"""One-view training step, in the same order as SplatTrainer::step."""

import logging
import math
from dataclasses import dataclass

import numpy as np

from .gpu import GRID_CELL, GRID_SHAPE, REDUCE, Rasterizer, read_buffer
from .population import apply_density, density_plan, write_ply

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Metrics:
    l1: float
    ssim: float
    psnr: float
    depth: float


class Trainer:
    def __init__(self, gpu, gauss, extent, width, height, views, hyper, budget, mean_steps, grid_steps):
        if min(width, height) < 16:
            raise ValueError("Training frames must be at least 16×16")
        self.gpu = gpu
        self.hyper = hyper
        self.budget = budget
        self.extent = extent
        self.mean_steps = mean_steps
        self.grid_steps = grid_steps
        self.gauss_step = 0
        self.grid_step = 0
        self.grid_view_steps = [0] * views
        self.raster = Rasterizer(gpu, width, height)
        self.shape = (height, width, 3)
        self.threads = (width, height, 1)
        pixels = width * height

        identity = np.zeros(GRID_SHAPE, dtype=np.float32)
        identity[[0, 5, 10]] = 1
        grids = np.tile(identity.reshape(-1), views)
        self.grid = gpu.buffer(grids.nbytes, "Bilateral grids", grids)
        self.grid_m = gpu.buffer(grids.nbytes, "Grid Adam first moments")
        self.grid_v = gpu.buffer(grids.nbytes, "Grid Adam second moments")
        self.grid_grad = gpu.buffer(grids.nbytes, "Grid gradients")
        self.clear_grid_grad = np.zeros(GRID_CELL, dtype=np.uint32)
        self.target = gpu.buffer(pixels * 12, "RGB target")
        self.depth_gt = gpu.buffer(pixels * 4, "Depth target")
        self.depth_grad = gpu.buffer(pixels * 4, "Depth gradient")
        self.corrected = gpu.buffer(pixels * 12, "Corrected RGB")
        self.dst_grad = gpu.buffer(pixels * 12, "Corrected RGB gradient")
        self.src_grad = gpu.buffer(pixels * 12, "Rendered RGB gradient")
        self.stats = gpu.buffer(pixels * 48, "SSIM stats and partial derivatives")
        self.color_partial = gpu.buffer(REDUCE * 3 * 4, "Color metric reduction")
        self.depth_partial = gpu.buffer(REDUCE * 2 * 4, "Depth metric reduction")
        self.capacity = 0
        self.set_population(gauss)
        command = gpu.encoder()
        command.clear_buffer(self.grid_m)
        command.clear_buffer(self.grid_v)
        gpu.submit(command)

    def set_population(self, values, moment=None, velocity=None):
        gpu = self.gpu
        self.count = len(values)
        allocated = self.count > self.capacity
        if allocated:
            self.gauss = gpu.buffer(self.count * 64, "Gaussians", values)
            self.grad = gpu.buffer(self.count * 64, "Gaussian gradients")
            self.moment = gpu.buffer(self.count * 64, "Gaussian Adam first moments", moment)
            self.velocity = gpu.buffer(self.count * 64, "Gaussian Adam second moments", velocity)
            self.density = gpu.buffer(self.count * 8, "Densification statistics")
            self.accum = gpu.buffer(self.count * 48, "Rasterizer gradients")
            self.raster.allocate_population(self.count)
            self.capacity = self.count
        command = gpu.encoder()
        command.clear_buffer(self.density)
        if moment is None:
            command.clear_buffer(self.moment)
            command.clear_buffer(self.velocity)
        if not allocated:
            command.upload_buffer_data(self.gauss, 0, values)
            if moment is not None:
                command.upload_buffer_data(self.moment, 0, moment)
                command.upload_buffer_data(self.velocity, 0, velocity)
        gpu.submit(command)

    def gaussians(self):
        return read_buffer(self.gauss, np.float32, (self.count, 16))

    def densify(self):
        if not self.hyper.density_step(self.gauss_step):
            return
        count = self.count
        values = self.gaussians()
        stats = read_buffer(self.density, np.float32, (count, 2))
        plan = density_plan(values, stats, self.gauss_step, self.budget, self.extent, self.hyper)
        keep, duplicate, split, reset = plan
        if len(keep) == count and not duplicate.any() and not reset:
            command = self.gpu.encoder()
            command.clear_buffer(self.density)
            self.gpu.submit(command)
            return

        moment = read_buffer(self.moment, np.float32, (count, 16))
        velocity = read_buffer(self.velocity, np.float32, (count, 16))
        self.set_population(*apply_density(values, moment, velocity, plan, self.gauss_step, self.hyper))
        logger.info(
            "Gaussian step %d: %d -> %d (%d cloned, %d split, %d pruned, opacity reset %s)",
            self.gauss_step, count, self.count, int((duplicate & ~split).sum()), int(split.sum()),
            count - len(keep), reset,
        )

    def slice_grid(self, command, grid_pc, backward=False):
        self.gpu.dispatch(
            command, "sliceGrid", self.threads, grid_pc | {"backward": int(backward)},
            grid=self.grid, src=self.raster.image, dst=self.corrected,
            dstGrad=self.dst_grad, srcGrad=self.src_grad, gridGrad=self.grid_grad,
        )

    def ssim_forward(self, command, pc, measure):
        gpu = self.gpu
        gpu.dispatch(
            command, "ssimForward", self.threads, pc,
            rendered=self.corrected, target=self.target, stats=self.stats,
        )
        if measure:
            gpu.dispatch(
                command, "reduceLoss", REDUCE, pc,
                target=self.target, stats=self.stats, corrected=self.corrected, partial=self.color_partial,
            )

    def step(self, view, index, rgb, depth, *, train):
        gpu = self.gpu
        h = self.hyper
        camera = self.raster.camera(view, self.count)
        with_depth = depth is not None
        self.raster.render(self.gauss, self.count, camera, with_depth)
        frame = self.raster.frame
        ssim = frame | {"scale": -h.ssimWeight / (self.raster.pixels * 3)}
        grid = frame | {"view": index, "views": len(self.grid_view_steps), "tv": h.tvLambda}
        command = gpu.encoder()
        command.upload_buffer_data(self.target, 0, rgb)
        if with_depth:
            command.upload_buffer_data(self.depth_gt, 0, depth)
            gpu.dispatch(
                command, "depthLoss", REDUCE, frame | {"depthMin": h.depthMin},
                pixels=self.raster.blend, depthGt=self.depth_gt, depthGrad=self.depth_grad,
                partial=self.depth_partial,
            )
        self.slice_grid(command, grid)
        # Train metrics use the render before the updates. Validation computes
        # the same gradients here and measures the post-update grid below.
        self.ssim_forward(command, ssim, measure=train)
        gpu.submit(command)

        depth_mean = depth_weight = 0.0
        if with_depth:
            depth_sum, hits = read_buffer(self.depth_partial, np.float32, (REDUCE, 2)).sum(
                axis=0, dtype=np.float64,
            )
            if hits:
                depth_mean = float(depth_sum / hits)
                depth_weight = self.extent * h.depthLambda / hits

        command = gpu.encoder()
        command.upload_buffer_data(self.grid_grad, index * GRID_CELL * 4, self.clear_grid_grad)
        gpu.dispatch(
            command, "ssimBackward", self.threads, ssim,
            rendered=self.corrected, target=self.target, stats=self.stats, imageGrad=self.dst_grad,
        )
        self.slice_grid(command, grid, backward=True)
        if train:
            command.clear_buffer(self.accum)
            self.raster.backward(command, self.accum, self.src_grad, self.depth_grad, depth_weight, with_depth)
            gpu.dispatch(
                command, "paramGrad", self.count,
                camera | {"collectDensity": int(self.gauss_step < h.densifyUntil)},
                gaussians=self.gauss, accum=self.accum, grads=self.grad,
                densify=self.density, tileCount=self.raster.counts,
            )
        gpu.dispatch(command, "gridTv", GRID_CELL, grid, grid=self.grid, gridGrad=self.grid_grad)

        adam = {"eps": h.adamEps, "beta1": h.beta1, "beta2": h.beta2}
        if train:
            self.gauss_step += 1
            gpu.dispatch(
                command, "adamGauss", self.count,
                adam | {
                    "count": self.count, "step": self.gauss_step,
                    "lrMean": h.mean_rate(self.gauss_step, self.mean_steps), "lrColor": h.lrColor,
                    "lrOp": h.lrOpacity, "lrQuat": h.lrQuat, "lrScale": h.lrScale,
                },
                value=self.gauss, grad=self.grad, moment=self.moment, vel=self.velocity,
            )
            if h.density_step(self.gauss_step):
                gpu.submit(command)
                self.densify()
                command = gpu.encoder()

        self.grid_step += 1
        self.grid_view_steps[index] += 1
        gpu.dispatch(
            command, "adamGrid", GRID_CELL,
            adam | {
                "offset": index * GRID_CELL, "count": GRID_CELL, "step": self.grid_view_steps[index],
                "lr": h.grid_rate(self.grid_step, self.grid_steps),
            },
            gridValue=self.grid, gridGrad=self.grid_grad, gridMoment=self.grid_m, gridVel=self.grid_v,
        )
        if not train:
            self.slice_grid(command, grid)
            self.ssim_forward(command, ssim, measure=True)
        gpu.submit(command)

        l1, mse, ssim_mean = read_buffer(self.color_partial, np.float32, (REDUCE, 3)).sum(
            axis=0, dtype=np.float64,
        ) / (self.raster.pixels * 3)
        psnr = -10 * math.log10(mse) if mse > 1e-12 else 100.0
        if not np.isfinite([l1, mse, ssim_mean, depth_mean]).all():
            raise RuntimeError(f"Non-finite metrics at frame {view.id}, grid step {self.grid_step}")
        return Metrics(float(l1), float(ssim_mean), psnr, depth_mean)

    def corrected_rgb(self):
        rgb = read_buffer(self.corrected, np.float32, self.shape)
        return (np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8)

    def export(self, path):
        values = self.gaussians()
        if not np.isfinite(values).all():
            raise RuntimeError("Cannot export non-finite Gaussian parameters")
        write_ply(path, values)
