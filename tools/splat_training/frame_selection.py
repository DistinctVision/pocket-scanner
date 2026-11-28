"""Remove redundant training views while preserving observed surface support."""

from dataclasses import dataclass
from math import cos, radians

import numpy as np
from PIL import Image

from .capture import Capture, View

DEPTH_MIN = 0.05
EDGE_INSET = 6
DEPTH_TOLERANCE_M = 0.04
DEPTH_TOLERANCE_RELATIVE = 0.015
PAIR_DISTANCE_M = 0.05
PAIR_ANGLE_DEGREES = 3
PATCH_VOXELS = 5
PATCH_COVERAGE_PERCENT = 92
PATCH_PARALLAX_PERCENT = 75
GLOBAL_COVERAGE_PERCENT = 99
GLOBAL_PARALLAX_PERCENT = 95
NEAR_POSITION_M = 0.35
NEAR_ROTATION_DEGREES = 18
PAIR_ANGLE_TERM = np.float32(2 * (1 - np.cos(np.deg2rad(PAIR_ANGLE_DEGREES))))
SAMPLE_BATCH_SIZE = 65_536
NEIGHBOR_Y, NEIGHBOR_X = np.indices((3, 3)).reshape(2, 9) - 1


@dataclass(frozen=True)
class FrameSelection:
    kept_indices: tuple[int, ...]
    dropped_indices: tuple[int, ...]
    unverified_indices: tuple[int, ...]
    covered_samples: int
    parallax_samples: int
    retained_covered_samples: int
    retained_parallax_samples: int


@dataclass(frozen=True)
class Observation:
    indices: np.ndarray
    distances: np.ndarray
    depth_agreement: float
    sharpness: float


def observe_depth(
    view: View, depth: np.ndarray, points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Project world samples and retain those supported by the measured Z16."""
    if len(points) == 0:
        return np.empty(0, dtype=np.intp), np.empty(0, dtype=np.float32), 0.0
    indices, distances = [], []
    comparable_count = 0
    fx, fy, cx, cy = view.intrinsics
    for start in range(0, len(points), SAMPLE_BATCH_SIZE):
        delta = points[start:start + SAMPLE_BATCH_SIZE] - view.pose[:3, 3]
        camera = delta @ view.pose[:3, :3]
        x, y, z = camera.T
        with np.errstate(invalid="ignore", divide="ignore"):
            px, py = fx * x / z + cx, fy * y / z + cy
        projected = np.flatnonzero(
            (z > DEPTH_MIN)
            & (px >= EDGE_INSET) & (px < view.width - EDGE_INSET)
            & (py >= EDGE_INSET) & (py < view.height - EDGE_INSET)
        )
        px = np.rint(px[projected]).astype(np.intp)
        py = np.rint(py[projected]).astype(np.intp)
        neighbors = depth[py[:, None] + NEIGHBOR_Y, px[:, None] + NEIGHBOR_X]
        valid = np.isfinite(neighbors) & (neighbors > DEPTH_MIN)
        counts = valid.sum(axis=1)
        ordered = np.sort(np.where(valid, neighbors, np.inf), axis=1)
        rows = np.arange(len(projected))
        # Five valid measurements are needed to corroborate the surface.
        median = ordered[rows, counts // 2]
        expected = z[projected]
        comparable = counts >= 5
        supported = comparable & (
            np.abs(expected - median) <= DEPTH_TOLERANCE_M + DEPTH_TOLERANCE_RELATIVE * expected
        )
        comparable_count += int(comparable.sum())
        confirmed = projected[supported]
        indices.append(start + confirmed)
        distances.append(np.linalg.norm(delta[confirmed], axis=1))
    sample_ids = np.concatenate(indices)
    return sample_ids, np.concatenate(distances), len(sample_ids) / max(comparable_count, 1)


def image_sharpness(image: Image.Image) -> float:
    gray = np.asarray(image.convert("L").resize((320, 240), Image.Resampling.BOX), dtype=np.float32)
    laplacian = (
        gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:]
        - 4 * gray[1:-1, 1:-1]
    )
    return float(np.quantile(np.abs(laplacian), 0.85))


def pair_samples(lookup: np.ndarray, other: Observation, baseline_squared: float) -> np.ndarray:
    left = lookup[other.indices]
    right = other.distances
    # Law of cosines with a cancellation-resistant minimum baseline term.
    required = np.square(left - right) + PAIR_ANGLE_TERM * left * right
    return other.indices[(left > 0) & (baseline_squared >= required)]


def prune_views(
    patches: np.ndarray, poses: np.ndarray, observations: list[Observation], order: list[int],
    *, near_position: float = NEAR_POSITION_M, near_rotation: float = NEAR_ROTATION_DEGREES,
) -> FrameSelection:
    count = len(observations)
    sample_count = len(patches)
    patch_count = int(patches.max()) + 1 if sample_count else 0
    lookup = np.zeros(sample_count, dtype=np.float32)
    covered = np.zeros(sample_count, dtype=np.int32)
    pairs = np.zeros(sample_count, dtype=np.int64)
    neighbors: list[list[tuple[int, float]]] = [[] for _ in observations]
    verified = np.array([len(observation.indices) > 0 for observation in observations], dtype=bool)
    # Every verified original view must still have a nearby representative.
    # The relative rotation includes roll as well as the viewing direction.
    positions = poses[:, :3, 3]
    rotations = poses[:, :3, :3].reshape(count, 9)
    nearby = np.zeros((count, count), dtype=bool)
    np.fill_diagonal(nearby, verified)
    near_rotation_trace = 1 + 2 * cos(radians(near_rotation))
    for i, observation in enumerate(observations):
        if not verified[i]:
            continue
        covered[observation.indices] += 1
        lookup[observation.indices] = observation.distances
        baselines = np.sum(np.square(positions[i + 1:] - positions[i]), axis=1)
        close = (baselines <= near_position ** 2) & (
            rotations[i + 1:] @ rotations[i] >= near_rotation_trace
        ) & verified[i + 1:]
        nearby[i, i + 1:] = nearby[i + 1:, i] = close
        for offset in np.flatnonzero((baselines >= PAIR_DISTANCE_M ** 2) & verified[i + 1:]):
            j = i + 1 + int(offset)
            baseline = float(baselines[offset])
            supported = pair_samples(lookup, observations[j], baseline)
            if len(supported):
                pairs[supported] += 1
                neighbors[i].append((j, baseline))
                neighbors[j].append((i, baseline))
        lookup[observation.indices] = 0

    patch_coverage = np.bincount(patches[covered > 0], minlength=patch_count)
    patch_parallax = np.bincount(patches[pairs > 0], minlength=patch_count)
    # Round allowable loss down. Small patches keep all their measured support.
    coverage_allowance = patch_coverage * (100 - PATCH_COVERAGE_PERCENT) // 100
    parallax_allowance = patch_parallax * (100 - PATCH_PARALLAX_PERCENT) // 100
    covered_samples = int(patch_coverage.sum())
    parallax_samples = int(patch_parallax.sum())
    remaining_coverage_loss = covered_samples * (100 - GLOBAL_COVERAGE_PERCENT) // 100
    remaining_parallax_loss = parallax_samples * (100 - GLOBAL_PARALLAX_PERCENT) // 100
    representatives = nearby.sum(axis=1)
    active = np.ones(count, dtype=bool)
    incident_pairs = np.zeros_like(pairs)
    for i in order:
        represented = nearby[i]
        if not verified[i] or np.any(representatives[represented] <= 1):
            continue
        observation = observations[i]
        samples = observation.indices
        uncovered = samples[covered[samples] == 1]
        if len(uncovered) > remaining_coverage_loss:
            continue
        coverage_loss = np.bincount(patches[uncovered], minlength=patch_count)
        if np.any(coverage_loss > coverage_allowance):
            continue

        lookup[samples] = observation.distances
        for j, baseline in neighbors[i]:
            if active[j]:
                incident_pairs[pair_samples(lookup, observations[j], baseline)] += 1
        lookup[samples] = 0

        incident = incident_pairs[samples]
        lost_parallax = samples[(incident > 0) & (incident == pairs[samples])]
        parallax_loss = np.bincount(patches[lost_parallax], minlength=patch_count)
        if len(lost_parallax) <= remaining_parallax_loss and np.all(parallax_loss <= parallax_allowance):
            active[i] = False
            covered[samples] -= 1
            pairs[samples] -= incident
            coverage_allowance -= coverage_loss
            parallax_allowance -= parallax_loss
            remaining_coverage_loss -= len(uncovered)
            remaining_parallax_loss -= len(lost_parallax)
            representatives[represented] -= 1
        incident_pairs[samples] = 0

    return FrameSelection(
        tuple(np.flatnonzero(active).tolist()),
        tuple(np.flatnonzero(~active).tolist()),
        tuple(np.flatnonzero(~verified).tolist()),
        covered_samples,
        parallax_samples,
        int(np.count_nonzero(covered)),
        int(np.count_nonzero(pairs)),
    )


def select_training_frames(
    capture: Capture, views: list[View], points: np.ndarray, voxel: float,
    *, near_position: float = NEAR_POSITION_M, near_rotation: float = NEAR_ROTATION_DEGREES,
) -> FrameSelection:
    if not views:
        return FrameSelection((), (), (), 0, 0, 0, 0)
    observations = []
    for view in views:
        indices, distances, agreement = observe_depth(view, capture.depth(view), points)
        sharpness = image_sharpness(capture.color(view)) if len(indices) else 0.0
        observations.append(Observation(indices, distances, agreement, sharpness))
    _, patches = np.unique(np.floor(points / (PATCH_VOXELS * voxel)), axis=0, return_inverse=True)
    order = sorted(
        range(len(views)),
        key=lambda i: (observations[i].depth_agreement, observations[i].sharpness, -views[i].id),
    )
    return prune_views(
        patches, np.stack([view.pose for view in views]), observations, order,
        near_position=near_position, near_rotation=near_rotation,
    )
