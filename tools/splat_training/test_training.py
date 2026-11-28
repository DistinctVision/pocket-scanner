"""Regression checks for density control and the SlangPy training pipeline."""

import unittest
from dataclasses import replace
from io import BytesIO
from struct import pack
from unittest.mock import patch

import numpy as np

from .capture import View, read_training_mesh
from .frame_selection import Observation, observe_depth, prune_views
from .gpu import ENTRIES, GRID_CELL, Gpu, Rasterizer, read_buffer
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
        poses = np.repeat(np.eye(4, dtype=np.float32)[None, :, :], len(positions), axis=0)
        poses[:, :3, 3] = positions
        result = prune_views(np.arange(len(points)), poses, observations, [2, 1, 0, 3, 4])
        self.assertEqual(result.kept_indices, (0, 2, 3, 4))
        self.assertEqual(result.dropped_indices, (1,))
        self.assertEqual(result.unverified_indices, (4,))
        self.assertEqual(result.covered_samples, 2)
        self.assertEqual(result.parallax_samples, 1)

    def test_coverage_loss_is_limited_globally_and_in_small_patches(self):
        patches = np.zeros(203, dtype=np.intp)
        patches[-1] = 1
        observed = ([0], [1], [2], [202], np.arange(3, 202))
        observations = [
            Observation(np.asarray(samples), np.ones(len(samples), dtype=np.float32), 1.0, 1.0)
            for samples in observed
        ]
        poses = np.repeat(np.eye(4, dtype=np.float32)[None, :, :], len(observations), axis=0)
        result = prune_views(patches, poses, observations, list(range(len(observations))))
        self.assertEqual(result.kept_indices, (2, 3, 4))
        self.assertEqual(result.retained_covered_samples, 201)

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
        with patch.dict(ENTRIES, {"ssim": (*ENTRIES["ssim"], "ssimForward", "ssimBackward")}):
            cls.gpu = Gpu("auto")

    def test_separable_ssim_matches_the_reference_forward_and_gradient(self):
        gpu = self.gpu
        shape = (17, 19, 3)
        height, width, _ = shape
        counts = (width, height, 1)
        rng = np.random.default_rng(13)
        a = rng.random(shape, dtype=np.float32)
        b = rng.random(shape, dtype=np.float32)
        rendered = gpu.buffer(a.nbytes, "Rendered", a)
        target = gpu.buffer(b.nbytes, "Target", b)
        stats = gpu.buffer(a.nbytes * 4, "Reference kernel statistics")
        fast_stats = gpu.buffer(a.nbytes * 4, "Separable kernel statistics")
        grad = gpu.buffer(a.nbytes, "Reference kernel gradient")
        fast_grad = gpu.buffer(a.nbytes, "Separable kernel gradient")
        rows = gpu.buffer(a.nbytes * 5, "Horizontal convolutions")
        pc = {"width": width, "height": height, "scale": -0.2 / a.size}
        command = gpu.encoder()
        gpu.dispatch(command, "ssimForward", counts, pc, rendered=rendered, target=target, stats=stats)
        gpu.dispatch(command, "ssimBackward", counts, pc, rendered=rendered, target=target, stats=stats, imageGrad=grad)
        gpu.dispatch(command, "ssimForwardRows", counts, pc, rendered=rendered, target=target, rowSums=rows)
        gpu.dispatch(command, "ssimForwardColumns", counts, pc, rowSums=rows, stats=fast_stats)
        gpu.dispatch(command, "ssimBackwardRows", counts, pc, rowSums=rows, stats=fast_stats)
        gpu.dispatch(command, "ssimBackwardColumns", counts, pc, rendered=rendered, target=target, rowSums=rows, imageGrad=fast_grad)
        gpu.submit(command)
        stat_shape = shape + (4,)
        np.testing.assert_allclose(
            read_buffer(fast_stats, np.float32, stat_shape),
            read_buffer(stats, np.float32, stat_shape), atol=2e-5, rtol=5e-4,
        )
        np.testing.assert_allclose(
            read_buffer(fast_grad, np.float32, shape), read_buffer(grad, np.float32, shape),
            atol=2e-8, rtol=5e-4,
        )

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
        self.assertAlmostEqual(trainer.raw_color_metrics()["raw_l1"], float(np.abs(rgb - rendered).mean()), places=6)
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

    def test_coarse_updates_keep_optimizer_state_for_full_resolution_validation(self):
        hyper = replace(Hyper(), lrGrid=0)
        seed = np.array([[0, 0, 1, 0, 0, 1, .4, .2, .1]], dtype=np.float32)
        gauss, extent = seed_gaussians(seed, .08, hyper)
        trainer = Trainer(self.gpu, gauss, extent, 16, 16, 1, hyper, 10, 2, 0)
        view = View(1, 32, 32, np.eye(4, dtype=np.float32), (64, 64, 16, 16), "", "")
        trainer.step(view, 0, np.zeros((16, 16, 3), dtype=np.float32), None, train=True)
        trained = trainer.gaussians()
        self.assertFalse(np.array_equal(trained, gauss))
        moment = read_buffer(trainer.moment, np.float32, gauss.shape)
        trainer.set_resolution(32, 32)
        rgb = np.full((32, 32, 3), .4, dtype=np.float32)
        depth = np.full((32, 32), 1.1, dtype=np.float32)
        depth[0, 0] = 0
        metrics = trainer.step(view, 0, rgb, depth, train=False)
        valid, matched, abs_sum, square_sum = trainer.validation_depth_totals(depth)
        self.assertEqual(valid, 1023)
        self.assertGreater(matched, 0)
        self.assertLess(matched, valid)
        self.assertGreater(abs_sum, 0)
        self.assertGreater(square_sum, 0)
        self.assertAlmostEqual(metrics.l1, trainer.raw_color_metrics()["raw_l1"], places=6)
        np.testing.assert_array_equal(trainer.gaussians(), trained)
        np.testing.assert_array_equal(read_buffer(trainer.moment, np.float32, gauss.shape), moment)
        self.assertEqual(trainer.gauss_step, 1)
        self.assertEqual(trainer.grid_step, 0)


if __name__ == "__main__":
    unittest.main()
