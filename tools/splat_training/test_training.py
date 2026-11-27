"""Regression checks for density control and the SlangPy training pipeline."""

import unittest
from dataclasses import replace
from io import BytesIO
from struct import pack

import numpy as np

from .capture import View, read_training_mesh
from .frame_selection import Observation, observe_depth, prune_views
from .gpu import GRID_CELL, Gpu, Rasterizer, read_buffer
from .population import Hyper, apply_density, density_plan, logit, seed_gaussians
from .trainer import Trainer


class FrameSelectionTest(unittest.TestCase):
    def test_depth_support_rejects_occluded_and_unknown_surface_samples(self):
        pose = np.eye(4, dtype=np.float32)
        pose[0, 3] = 0.5
        view = View(1, 32, 32, pose, (20, 20, 16, 16), "", "")
        depth = np.ones((32, 32), dtype=np.float32)
        depth[15:18, 22:25] = 0
        points = np.array(
            [[.5, 0, 1], [.5, 0, 1.3], [.85, 0, 1], [3, 0, 1]], dtype=np.float32,
        )
        indices, distances, agreement = observe_depth(view, depth, points)
        np.testing.assert_array_equal(indices, [0])
        np.testing.assert_array_equal(distances, [1])
        self.assertEqual(agreement, 0.5)

    def test_redundancy_preserves_the_only_translated_view_and_unverified_views(self):
        points = np.array([[0, 0, 1], [0, .2, 1]], dtype=np.float32)
        positions = np.array([[0, 0, 0], [0, 0, 0], [.12, 0, 0], [0, .2, 0], [0, 0, 0]], dtype=np.float32)
        observed = ([0], [0], [0], [1], [])
        observations = [
            Observation(
                np.array(indices, dtype=np.intp),
                np.linalg.norm(points[indices] - positions[i], axis=1), 1.0, 1.0,
            )
            for i, indices in enumerate(observed)
        ]
        result = prune_views(len(points), positions, observations, [2, 1, 0, 3, 4])
        self.assertEqual(result.kept_indices, (0, 2, 3, 4))
        self.assertEqual(result.dropped_indices, (1,))
        self.assertEqual(result.unverified_indices, (4,))
        self.assertEqual(result.covered_samples, 2)
        self.assertEqual(result.parallax_samples, 1)

    def test_coverage_samples_are_independent_of_the_gaussian_budget(self):
        header = (
            b"ply\nformat binary_little_endian 1.0\ncomment voxel 0.08\n"
            b"element vertex 4\nproperty float x\nproperty float y\nproperty float z\n"
            b"property uchar red\nproperty uchar green\nproperty uchar blue\n"
            b"element face 2\nproperty list uchar int vertex_indices\nend_header\n"
        )
        vertices = b"".join(pack("<3f3B", x, y, 1, 0, 0, 0) for x, y in ((0, 0), (1, 0), (0, 1), (1, 1)))
        faces = pack("<B3iB3i", 3, 0, 1, 2, 3, 1, 3, 2)
        mesh = read_training_mesh(BytesIO(header + vertices + faces), 3, include_coverage=True)
        self.assertEqual(len(mesh.seed), 3)
        self.assertEqual(len(mesh.coverage_points), 6)
        np.testing.assert_array_equal(mesh.coverage_points[3], [1, 1, 1])
        np.testing.assert_allclose(mesh.coverage_points[4:], [[1/3, 1/3, 1], [2/3, 2/3, 1]])


class DensityTest(unittest.TestCase):
    def test_prune_clone_split_and_reset_preserve_the_right_moments(self):
        hyper = replace(Hyper(), minKeep=2, opacityResetEvery=1)
        gauss = np.zeros((5, 16), dtype=np.float32)
        gauss[:, 11] = 1
        gauss[:, 0] = np.arange(5)
        gauss[:, 12:15] = np.log(np.array([.005, .04, .4, .005, .005]))[:, None]
        gauss[3, 3] = -10
        stats = np.array([[.01, 1], [.01, 1], [.01, 1], [.01, 1], [0, 1]], dtype=np.float32)
        moment = np.broadcast_to(np.arange(5, dtype=np.float32)[:, None] + 1, gauss.shape)
        plan = density_plan(gauss, stats, 500, 5, 1.0, hyper)
        values, m, v = apply_density(gauss, moment, moment, plan, 500, hyper)
        self.assertEqual(len(values), 5)
        np.testing.assert_array_equal(values[[0, 1, 4], 0], [0, 0, 4])
        np.testing.assert_allclose(values[2:4, 12:15], np.log(.04 / 1.6), rtol=1e-6)
        self.assertGreater(np.linalg.norm(values[2, :3] - values[3, :3]), 0)
        np.testing.assert_allclose(values[:, 3], logit(.01))
        np.testing.assert_array_equal(m[:, 0], [1, 0, 0, 0, 5])
        np.testing.assert_array_equal(m[:, 3], np.zeros(5))
        np.testing.assert_array_equal(m, v)


class SlangTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gpu = Gpu("auto")

    def test_gpu_radix_sort_is_stable_across_workgroups(self):
        raster = Rasterizer(self.gpu, 80, 48)
        count = 20_003  # Multi-level scan, partial trailing group, repeated tile/depth keys.
        rng = np.random.default_rng(123)
        keys = rng.integers([0, 0], [11, raster.tiles], size=(count, 2), dtype=np.uint32)
        keys[:, 0] = keys[:, 0].astype(np.float32).view(np.uint32)
        ids = np.arange(count, dtype=np.uint32)
        raster.allocate_pairs(count)
        raster.key_a.copy_from_numpy(keys)
        raster.id_a.copy_from_numpy(ids)
        sorted_keys, sorted_ids = raster.sort_pairs(count)
        expected = np.lexsort((keys[:, 0], keys[:, 1]))
        np.testing.assert_array_equal(read_buffer(sorted_ids, np.uint32, (count,)), expected)
        np.testing.assert_array_equal(read_buffer(sorted_keys, np.uint32, (count, 2)), keys[expected])

    def test_validation_updates_only_its_grid_and_measures_after_adam(self):
        hyper = Hyper()
        seed = np.array([
            [-.02, 0, 1, 0, 0, 1, .4, .2, .1],
            [0, 0, 1, 0, 0, 1, .3, .6, .2],
            [.02, 0, 1, 0, 0, 1, .1, .2, .7],
        ], dtype=np.float32)
        gauss, extent = seed_gaussians(seed, .08, hyper)
        width, height = 19, 17
        trainer = Trainer(self.gpu, gauss, extent, width, height, 2, hyper, 20, 2, 4)
        view = View(1, width, height, np.eye(4, dtype=np.float32), (64, 64, 9, 8), "", "")
        rgb = np.full((height, width, 3), .4, dtype=np.float32)
        depth = np.full((height, width), 1.1, dtype=np.float32)

        before_grid = read_buffer(trainer.grid, np.float32, (2, GRID_CELL))
        val = trainer.step(view, 1, rgb, depth, train=False)
        rendered = read_buffer(trainer.raster.image, np.float32, rgb.shape)
        corrected = read_buffer(trainer.corrected, np.float32, rgb.shape)
        self.assertGreater(trainer.raster.pair_count, 0)
        self.assertGreater(val.depth, .08)
        self.assertLess(val.l1, np.abs(rgb - rendered).mean())
        self.assertAlmostEqual(val.l1, float(np.abs(rgb - corrected).mean()), places=6)
        np.testing.assert_array_equal(trainer.gaussians(), gauss)
        after_grid = read_buffer(trainer.grid, np.float32, (2, GRID_CELL))
        np.testing.assert_array_equal(before_grid[0], after_grid[0])
        self.assertFalse(np.array_equal(before_grid[1], after_grid[1]))
        self.assertEqual(trainer.grid_view_steps, [0, 1])
        self.assertEqual(trainer.gauss_step, 0)

        train = trainer.step(view, 0, rgb, depth, train=True)
        self.assertAlmostEqual(train.l1, float(np.abs(rgb - rendered).mean()), places=6)
        self.assertFalse(np.array_equal(trainer.gaussians(), gauss))
        self.assertEqual(trainer.grid_view_steps, [1, 1])
        self.assertEqual(trainer.gauss_step, 1)


if __name__ == "__main__":
    unittest.main()
