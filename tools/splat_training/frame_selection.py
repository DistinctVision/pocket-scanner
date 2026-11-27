"""Remove redundant training views while preserving observed surface support."""

from dataclasses import dataclass

import numpy as np
from PIL import Image

from .capture import Capture, View

DEPTH_MIN = 0.05
EDGE_INSET = 6
DEPTH_TOLERANCE_M = 0.04
DEPTH_TOLERANCE_RELATIVE = 0.015
PAIR_DISTANCE_M = 0.05
PAIR_ANGLE_DEGREES = 3
POSE_POSITION_STEP_M = 0.14
POSE_DIRECTION_STEP = 0.12
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


def removal_order(views: list[View], observations: list[Observation]) -> list[int]:
    positions = np.stack([view.pose[:3, 3] for view in views])
    forwards = np.stack([view.pose[:3, 2] for view in views])
    lengths = np.linalg.norm(forwards, axis=1, keepdims=True)
    np.divide(forwards, lengths, out=forwards, where=lengths > 0)
    scaled = np.column_stack((positions / POSE_POSITION_STEP_M, forwards / POSE_DIRECTION_STEP))
    buckets = np.copysign(np.floor(np.abs(scaled) + 0.5), scaled).astype(np.int64)

    # Prefer better Z16 alignment and then sharper images within a group of
    # similar poses. Unique measured support always overrides this ordering.
    ranked = sorted(
        range(len(views)),
        key=lambda i: (observations[i].depth_agreement, observations[i].sharpness, -views[i].id),
    )
    best_by_pose = {tuple(buckets[i]): i for i in ranked}
    best = set(best_by_pose.values())
    return [i for i in ranked if i not in best] + [i for i in ranked if i in best]


def pair_samples(lookup: np.ndarray, other: Observation, baseline_squared: float) -> np.ndarray:
    left = lookup[other.indices]
    right = other.distances
    # Law of cosines with a cancellation-resistant minimum baseline term.
    required = np.square(left - right) + PAIR_ANGLE_TERM * left * right
    return other.indices[(left > 0) & (baseline_squared >= required)]


def prune_views(
    sample_count: int, positions: np.ndarray, observations: list[Observation], order: list[int],
) -> FrameSelection:
    count = len(observations)
    lookup = np.zeros(sample_count, dtype=np.float32)
    covered = np.zeros(sample_count, dtype=np.int32)
    pairs = np.zeros(sample_count, dtype=np.int64)
    neighbors: list[list[tuple[int, float]]] = [[] for _ in observations]
    unverified = []
    for i, observation in enumerate(observations):
        if len(observation.indices) == 0:
            unverified.append(i)
            continue
        covered[observation.indices] += 1
        lookup[observation.indices] = observation.distances
        baselines = np.sum(np.square(positions[i + 1:] - positions[i]), axis=1)
        for offset in np.flatnonzero(baselines >= PAIR_DISTANCE_M ** 2):
            j = i + 1 + int(offset)
            baseline = float(baselines[offset])
            supported = pair_samples(lookup, observations[j], baseline)
            if len(supported):
                pairs[supported] += 1
                neighbors[i].append((j, baseline))
                neighbors[j].append((i, baseline))
        lookup[observation.indices] = 0

    covered_samples = int(np.count_nonzero(covered))
    parallax_samples = int(np.count_nonzero(pairs))
    minimum_views = np.minimum(covered, 2)
    active = np.ones(count, dtype=bool)
    lost_pairs = np.zeros_like(pairs)
    for i in order:
        observation = observations[i]
        samples = observation.indices
        if len(samples) == 0 or np.any(covered[samples] <= minimum_views[samples]):
            continue

        lookup[samples] = observation.distances
        for j, baseline in neighbors[i]:
            if active[j]:
                lost_pairs[pair_samples(lookup, observations[j], baseline)] += 1
        lookup[samples] = 0

        lost = lost_pairs[samples]
        if not np.any((lost > 0) & (lost == pairs[samples])):
            active[i] = False
            covered[samples] -= 1
            pairs[samples] -= lost
        lost_pairs[samples] = 0

    return FrameSelection(
        tuple(np.flatnonzero(active).tolist()),
        tuple(np.flatnonzero(~active).tolist()),
        tuple(unverified),
        covered_samples,
        parallax_samples,
    )


def select_training_frames(capture: Capture, views: list[View], points: np.ndarray) -> FrameSelection:
    if not views:
        return FrameSelection((), (), (), 0, 0)
    observations = []
    for view in views:
        indices, distances, agreement = observe_depth(view, capture.depth(view), points)
        sharpness = image_sharpness(capture.color(view)) if len(indices) else 0.0
        observations.append(Observation(indices, distances, agreement, sharpness))
    return prune_views(
        len(points),
        np.stack([view.pose[:3, 3] for view in views]),
        observations,
        removal_order(views, observations),
    )
