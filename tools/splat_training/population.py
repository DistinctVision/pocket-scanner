"""Host-side initialization and density control from splat_train.cpp."""

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Hyper:
    lrMean: float = 1.6e-4
    lrMeanEnd: float = 0.01
    lrScale: float = 5e-3
    lrQuat: float = 1e-3
    lrOpacity: float = 5e-2
    lrColor: float = 2.5e-3
    lrGrid: float = 2e-3
    lrGridStart: float = 0.01
    lrGridEnd: float = 0.01
    gridWarmMax: int = 1000
    gridWarmDivisor: int = 5
    beta1: float = 0.9
    beta2: float = 0.999
    adamEps: float = 1e-15
    ssimWeight: float = 0.2
    tvLambda: float = 0.01
    depthLambda: float = 1e-2
    depthMin: float = 0.05
    densifyFrom: int = 500
    densifyUntil: int = 15000
    densifyInterval: int = 100
    gradThreshold: float = 2e-4
    percentDense: float = 0.01
    pruneScale: float = 0.1
    minOpacity: float = 0.005
    splitFactor: float = 1.6
    opacityResetEvery: int = 3000
    opacityReset: float = 0.01
    minKeep: int = 64
    seedOpacity: float = 0.5
    seedContour: float = 0.75
    seedThin: float = 0.1
    extentFloor: float = 1.0

    def mean_rate(self, step, horizon):
        return self.lrMean * self.lrMeanEnd ** min(1.0, step / horizon)

    def grid_rate(self, step, horizon):
        warm = max(1, min(self.gridWarmMax, horizon // max(1, self.gridWarmDivisor)))
        if step <= warm:
            return self.lrGrid * (self.lrGridStart + (1 - self.lrGridStart) * step / warm)
        return self.lrGrid * self.lrGridEnd ** min(1.0, (step - warm) / max(1, horizon - warm))

    def density_step(self, step):
        return (self.densifyFrom <= step <= self.densifyUntil
                and self.densifyInterval > 0 and step % self.densifyInterval == 0)


def logit(opacity):
    opacity = min(1 - 1e-4, max(1e-4, opacity))
    return math.log(opacity / (1 - opacity))


def seed_gaussians(seed, voxel, hyper):
    # Four float4 fields in the order of the Slang Gauss struct.
    gauss = np.zeros((len(seed), 16), dtype=np.float32)
    gauss[:, :3] = seed[:, :3]
    gauss[:, 3] = logit(hyper.seedOpacity)
    gauss[:, 4:7] = seed[:, 6:9]
    gauss[:, 8] = -seed[:, 4]
    gauss[:, 9] = seed[:, 3]
    gauss[:, 11] = 1 + seed[:, 5]
    gauss[gauss[:, 11] < 1e-4, 8:12] = (1, 0, 0, 0)
    quat = gauss[:, 8:12]
    quat /= np.linalg.norm(quat, axis=1)[:, None]
    opacity = min(1 - 1e-4, max(1e-4, hyper.seedOpacity))
    sigma = max(voxel * hyper.seedContour / math.sqrt(2 * math.log(255 * opacity)), 1e-4)
    gauss[:, 12:14] = math.log(sigma)
    gauss[:, 14] = math.log(max(sigma * hyper.seedThin, 1e-4))
    extent = max(hyper.extentFloor, float(np.linalg.norm(np.ptp(gauss[:, :3], axis=0))))
    return gauss, extent


def split_offsets(parent, indices, step):
    # Same per-parent uint32 LCG and Box–Muller draws as the Android trainer.
    rng = np.uint32(0x9E3779B9) ^ (indices.astype(np.uint32) * np.uint32(747796405)) ^ np.uint32(step)
    local = np.empty((len(parent), 2, 3), dtype=np.float32)
    for child in range(2):
        for axis in range(3):
            rng = rng * np.uint32(1664525) + np.uint32(1013904223)
            u1 = np.maximum((rng >> 8).astype(np.float32) / np.float32(16777216), 1e-6)
            rng = rng * np.uint32(1664525) + np.uint32(1013904223)
            u2 = (rng >> 8).astype(np.float32) / np.float32(16777216)
            local[:, child, axis] = np.sqrt(-2 * np.log(u1)) * np.cos(np.float32(6.2831853) * u2)
    local *= np.exp(parent[:, None, 12:15])
    xyz = parent[:, None, 8:11]
    w = parent[:, None, 11:12]
    return local + 2 * np.cross(xyz, np.cross(xyz, local) + w * local)


def density_plan(gauss, stats, step, budget, extent, hyper):
    """Return survivor indices in Adam order, with clone/split masks per survivor."""
    scale = np.exp(gauss[:, 12:15]).max(axis=1)
    opacity = 1 / (1 + np.exp(-np.clip(gauss[:, 3], -20, 20)))
    small = scale <= hyper.pruneScale * extent
    opaque = opacity >= hyper.minOpacity
    keep = np.flatnonzero(small & opaque)
    if len(keep) < hyper.minKeep:
        keep = np.flatnonzero(small)
        if len(keep) < hyper.minKeep:
            keep = np.arange(len(gauss))
    available = budget - len(keep)
    duplicate = np.zeros(len(keep), dtype=bool)
    if available > 0:
        sums, hits = stats[keep].T
        mean = np.divide(sums, hits, out=np.zeros_like(sums), where=hits > 0)
        eligible = np.flatnonzero(np.isfinite(mean) & (mean > hyper.gradThreshold))
        duplicate[eligible[:available]] = True
    split = duplicate & (scale[keep] > hyper.percentDense * extent)
    reset = hyper.opacityResetEvery > 0 and step % hyper.opacityResetEvery == 0
    return keep, duplicate, split, reset


def apply_density(gauss, moment, velocity, plan, step, hyper):
    keep, duplicate, split, reset = plan
    repeats = 1 + duplicate.astype(np.int64)
    first = np.cumsum(repeats) - repeats
    rows = np.repeat(keep, repeats)
    values, moment, velocity = gauss[rows], moment[rows], velocity[rows]
    children = first[split]
    offsets = split_offsets(gauss[keep[split]], keep[split], step)
    values[children, :3] += offsets[:, 0]
    values[children + 1, :3] += offsets[:, 1]
    shrink = math.log(max(hyper.splitFactor, 1.01))
    values[children, 12:15] -= shrink
    values[children + 1, 12:15] -= shrink
    fresh = np.concatenate((first[duplicate] + 1, children))
    moment[fresh] = 0
    velocity[fresh] = 0
    if reset:
        np.minimum(values[:, 3], logit(hyper.opacityReset), out=values[:, 3])
        moment[:, 3] = 0
        velocity[:, 3] = 0
    return values, moment, velocity


def write_ply(path: Path, gauss):
    names = (
        "x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
        "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
    )
    rows = np.zeros((len(gauss), len(names)), dtype="<f4")
    rows[:, :3] = gauss[:, :3]
    rows[:, 6:9] = (gauss[:, 4:7] - .5) / np.float32(.28209479177387814)
    rows[:, 9] = gauss[:, 3]
    rows[:, 10:13] = gauss[:, 12:15]
    rows[:, 13:17] = gauss[:, (11, 8, 9, 10)]
    tmp = path.with_suffix(".ply.tmp")
    with tmp.open("wb") as stream:
        stream.write(f"ply\nformat binary_little_endian 1.0\nelement vertex {len(rows)}\n".encode())
        for name in names:
            stream.write(f"property float {name}\n".encode())
        stream.write(b"end_header\n")
        rows.tofile(stream)
    tmp.replace(path)
