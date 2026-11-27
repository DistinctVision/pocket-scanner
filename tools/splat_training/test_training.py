"""Regression checks for density control and the SlangPy training pipeline."""

import unittest
from dataclasses import replace

import numpy as np

from .capture import View
from .gpu import GRID_CELL, Gpu, Rasterizer, read_buffer
from .population import Hyper, apply_density, density_plan, logit, seed_gaussians
from .trainer import Trainer


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
