"""Read Pocket Scanner's saved camera views and mesh seed."""

import json
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class View:
    id: int
    width: int
    height: int
    pose: np.ndarray
    intrinsics: tuple[float, float, float, float]
    color: str
    depth: str


@dataclass(frozen=True)
class TrainingMesh:
    seed: np.ndarray
    voxel: float
    coverage_points: np.ndarray | None


class Capture:
    def __init__(self, archive: ZipFile):
        self.archive = archive
        manifest = json.loads(archive.read("manifest.json"))
        if manifest.get("format") != "pocket-scan" or manifest.get("version") != 1:
            raise ValueError("Expected a Pocket Scanner v1 .pscan package")
        self.depth_scale = np.float32(manifest["depthScale"])
        if not np.isfinite(self.depth_scale) or self.depth_scale <= 0:
            raise ValueError("Invalid .pscan depthScale")
        self.views = []
        for frame in sorted(manifest["frames"], key=lambda frame: frame["id"]):
            if frame["id"] >= 1_000_000_000:  # Mesh-only coverage frame.
                continue
            pose = np.asarray(frame["pose"], dtype=np.float32).reshape(4, 4)
            intrinsics = tuple(float(frame[key]) for key in ("fx", "fy", "cx", "cy"))
            if not np.isfinite(pose).all() or not np.isfinite(intrinsics).all():
                raise ValueError(f"Invalid camera calibration for frame {frame['id']}")
            self.views.append(View(
                frame["id"], frame["width"], frame["height"], pose, intrinsics,
                frame["color"], frame["depth"],
            ))
        if not self.views:
            raise ValueError("The package has no training frames")
        if len({view.id for view in self.views}) != len(self.views):
            raise ValueError("Duplicate frame IDs in .pscan manifest")
        self.width, self.height = self.views[0].width, self.views[0].height
        self.views = [
            view for view in self.views if (view.width, view.height) == (self.width, self.height)
        ]

    def color(self, view: View) -> Image.Image:
        with self.archive.open(view.color) as stream, Image.open(stream) as image:
            if image.size != (view.width, view.height):
                raise ValueError(f"Unexpected PNG dimensions for frame {view.id}")
            return image.convert("RGB")

    def depth(self, view: View) -> np.ndarray:
        raw = self.archive.read(view.depth)
        if len(raw) != view.height * view.width * 2:
            raise ValueError(f"Unexpected Z16 size for frame {view.id}")
        depth = np.frombuffer(raw, dtype="<u2").astype(np.float32) / self.depth_scale
        return depth.reshape(view.height, view.width)

    def images(self, view: View) -> tuple[np.ndarray, np.ndarray]:
        rgb = np.asarray(self.color(view), dtype=np.float32) / np.float32(255)
        return rgb, self.depth(view)

    def seed(self, budget: int, mesh: Path | None = None):
        loaded = self.training_mesh(budget, mesh)
        return loaded.seed, loaded.voxel

    def training_mesh(
        self, budget: int, mesh: Path | None = None, *, include_coverage: bool = False,
    ) -> TrainingMesh:
        if mesh is not None:
            with mesh.open("rb") as stream:
                return read_training_mesh(stream, budget, include_coverage=include_coverage)
        if "mesh.ply" not in self.archive.namelist():
            raise ValueError("The package has no mesh.ply. Export the mesh in Android and pass --mesh.")
        with self.archive.open("mesh.ply") as stream:
            return read_training_mesh(stream, budget, include_coverage=include_coverage)


def mesh_seed(stream, budget: int):
    mesh = read_training_mesh(stream, budget)
    return mesh.seed, mesh.voxel


def read_training_mesh(stream, budget: int, *, include_coverage: bool = False) -> TrainingMesh:
    """Port of Volume::readMeshSeed and MeshSeed for the app's binary PLY."""
    if stream.readline().strip() != b"ply":
        raise ValueError("Expected a Pocket Scanner PLY mesh")
    vertex_count = face_count = 0
    voxel = 0.0
    binary = False
    for line in stream:
        fields = line.split()
        if fields == [b"end_header"]:
            break
        if fields == [b"format", b"binary_little_endian", b"1.0"]:
            binary = True
        elif fields[:2] == [b"comment", b"voxel"] and len(fields) == 3:
            voxel = float(np.float32(float(fields[2])))
        elif fields[:2] == [b"element", b"vertex"] and len(fields) == 3:
            vertex_count = int(fields[2])
        elif fields[:2] == [b"element", b"face"] and len(fields) == 3:
            face_count = int(fields[2])
    else:
        raise ValueError("Incomplete PLY header")
    if not binary or vertex_count < 3 or face_count < 1 or not np.isfinite(voxel) or voxel < float(np.float32(.01)):
        raise ValueError("Expected a binary Pocket Scanner mesh with 'comment voxel'")

    vertex_dtype = np.dtype([("xyz", "<f4", 3), ("rgb", "u1", 3)])
    face_dtype = np.dtype([("size", "u1"), ("indices", "<i4", 3)])
    vertex_bytes = stream.read(vertex_count * vertex_dtype.itemsize)
    face_bytes = stream.read(face_count * face_dtype.itemsize)
    if len(vertex_bytes) != vertex_count * 15 or len(face_bytes) != face_count * 13:
        raise ValueError("Truncated Pocket Scanner mesh")
    vertices = np.frombuffer(vertex_bytes, dtype=vertex_dtype)
    faces = np.frombuffer(face_bytes, dtype=face_dtype)
    if (faces["size"] != 3).any():
        raise ValueError("The Pocket Scanner mesh must contain triangles")
    triangles = faces["indices"]
    triangles = triangles[((triangles >= 0) & (triangles < vertex_count)).all(axis=1)]
    xyz = vertices["xyz"][triangles]
    normals = np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0])
    length = np.linalg.norm(normals.astype(np.float64), axis=1).astype(np.float32)
    valid = np.isfinite(xyz).all(axis=(1, 2)) & np.isfinite(length) & (length > 1e-8)
    xyz = xyz[valid].reshape(-1, 3)
    normals = np.repeat(normals[valid] / length[valid, None], 3, axis=0)
    colors = vertices["rgb"][triangles[valid]].reshape(-1, 3).astype(np.float32) / np.float32(255)

    scaled = xyz.astype(np.float64) * (1.0 / max(1e-6, voxel * 1e-3))
    # std::round rounds ties away from zero. Keep first encounter order for the budget.
    keys = np.copysign(np.floor(np.abs(scaled) + 0.5), scaled)
    _, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    order = first.argsort()
    selected = order[:budget]
    count = len(selected)
    if count < 3:
        raise ValueError("The mesh has fewer than three usable seed vertices")
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    index = rank[inverse]
    keep = index < count
    seed = np.zeros((count, 9), dtype=np.float32)
    seed[:, :3] = xyz[first[selected]]
    np.add.at(seed[:, 3:6], index[keep], normals[keep])
    np.add.at(seed[:, 6:9], index[keep], colors[keep])
    seed[:, 6:9] /= np.bincount(index[keep], minlength=count).astype(np.float32)[:, None]
    length = np.linalg.norm(seed[:, 3:6].astype(np.float64), axis=1).astype(np.float32)
    valid = length > 1e-8
    seed[valid, 3:6] /= length[valid, None]
    seed[~valid, 3:6] = (0, 0, 1)
    coverage = None
    if include_coverage:
        # Evaluate the measured surface before the Gaussian budget truncates it.
        centers = xyz.reshape(-1, 3, 3).mean(axis=1)
        coverage = np.concatenate((xyz[first[order]], centers))
    return TrainingMesh(seed, voxel, coverage)
