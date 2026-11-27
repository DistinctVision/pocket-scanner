"""SlangPy dispatch and tile rasterization using the Android shader sources."""

import re
import sys
from math import prod
from pathlib import Path

import numpy as np
import slangpy as spy

SHADERS = Path(__file__).resolve().parents[2] / "android/app/src/main/cpp/vk/shaders"
TILE = 16
SCAN = 128
REDUCE = 256
GRID_SHAPE = (12, 8, 16, 16)
GRID_CELL = prod(GRID_SHAPE)
BUFFER_USAGE = (
    spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access
    | spy.BufferUsage.copy_source | spy.BufferUsage.copy_destination
)
ENTRIES = {
    "splat_project": ("project",),
    "splat_tile": ("countTiles",),
    "splat_scan": ("scanBlock",),
    "splat_scan_add": ("scanAdd",),
    "splat_keys": ("writeKeys",),
    "splat_sort": ("radixHist",),
    "splat_scatter": ("radixScatter",),
    "splat_ranges": ("markRanges",),
    "splat_blend": ("blendRgb", "blendRgbEd"),
    "splat_blend_back": ("blendBackwardRgb", "blendBackwardRgbEd"),
    "splat_depth": ("depthLoss",),
    "splat_params": ("paramGrad",),
    "splat_adam": ("adamGauss",),
    "splat_adam_grid": ("adamGrid",),
    "bilagrid": ("sliceGrid", "gridTv"),
    "ssim": ("ssimForward", "ssimBackward", "reduceLoss"),
}


def ceil_div(n, block):
    return (n + block - 1) // block


def read_buffer(buffer, dtype, shape):
    size = prod(shape)
    return buffer.to_numpy().view(dtype).reshape(-1)[:size].reshape(shape)


class Gpu:
    def __init__(self, backend, shader_dir=SHADERS):
        if backend == "auto":
            backend = "metal" if sys.platform == "darwin" else "vulkan"
        self.backend = backend
        self.device = spy.Device(
            type=getattr(spy.DeviceType, backend),
            enable_hot_reload=False,
            compiler_options={
                "matrix_layout": spy.SlangMatrixLayout.row_major,
                "include_paths": [str(shader_dir)],
            },
        )
        self.kernels = {}
        for file, entries in ENTRIES.items():
            path = shader_dir / f"{file}.slang"
            source = path.read_text()
            # Use SlangPy's reflected bindings on both backends. Only the
            # Vulkan descriptor and push-constant declarations are adapted.
            source = re.sub(r"\[\[vk::binding\(\d+, 0\)\]\]\s*", "", source)
            source, fields = re.subn(
                r"\[\[vk::push_constant\]\]\s*(\w+) pc;", r"ConstantBuffer<\1> pc;", source,
            )
            if fields != 1:
                raise ValueError(f"{path.name} must declare a single push block named pc")
            module = self.device.load_module_from_source(file, source, path)
            for entry in entries:
                program = self.device.link_program([module], [module.entry_point(entry)])
                self.kernels[entry] = self.device.create_compute_kernel(program)

    def buffer(self, size, label, data=None):
        return self.device.create_buffer(size=size, label=label, data=data, usage=BUFFER_USAGE)

    def encoder(self):
        return self.device.create_command_encoder()

    def submit(self, command):
        self.device.submit_command_buffer(command.finish())

    def dispatch(self, command, entry, threads, pc, **buffers):
        if isinstance(threads, int):
            threads = (threads, 1, 1)
        self.kernels[entry].dispatch(
            thread_count=threads, vars={"pc": pc, **buffers}, command_encoder=command,
        )


class Rasterizer:
    def __init__(self, gpu, width, height):
        self.gpu = gpu
        self.width = width
        self.height = height
        self.tiles_x = ceil_div(width, TILE)
        self.tiles_y = ceil_div(height, TILE)
        self.tiles = self.tiles_x * self.tiles_y
        self.pixels = width * height
        self.frame = {"width": width, "height": height}
        self.tile = self.frame | {"tilesX": self.tiles_x, "tilesY": self.tiles_y}

        self.image = gpu.buffer(self.pixels * 12, "RGB render")
        self.blend = gpu.buffer(self.pixels * 12, "Per-pixel transmittance, end, expected depth")
        self.ranges = gpu.buffer(self.tiles * 8, "Tile ranges")
        self.tail = gpu.buffer(8, "Last tile offset and count")
        self.totals = gpu.buffer(16 * 4, "Radix digit totals")
        self.bases = gpu.buffer(16 * 4, "Radix digit offsets")
        self.scan_sums = {}
        self.pair_capacity = 0
        self.pair_count = 0

    def allocate_population(self, count):
        self.splats = self.gpu.buffer(count * 48, "Projected splats")
        self.counts = self.gpu.buffer(count * 4, "Tiles per splat")
        self.offsets = self.gpu.buffer(count * 4, "Exclusive tile offsets")

    def camera(self, view, count):
        return self.frame | {
            "count": count,
            "row0": spy.float4(*view.pose[0]),
            "row1": spy.float4(*view.pose[1]),
            "row2": spy.float4(*view.pose[2]),
            "intr": spy.float4(*view.intrinsics),
        }

    def scan(self, command, src, dst, count, base=0, level=0):
        """Hierarchical exclusive scan with the app's block and add kernels."""
        groups = ceil_div(count, SCAN)
        scratch = self.scan_sums.get(level)
        if scratch is None or scratch[0].size < groups * 4:
            scratch = (
                self.gpu.buffer(groups * 4, f"Scan sums {level}"),
                self.gpu.buffer(groups * 4, f"Scanned sums {level}"),
            )
            self.scan_sums[level] = scratch
        sums, prefixes = scratch
        self.gpu.dispatch(
            command, "scanBlock", count,
            {"n": count, "inBase": base, "outBase": base, "sumBase": 0},
            input=src, output=dst, sums=sums,
        )
        if groups > 1:
            self.scan(command, sums, prefixes, groups, level=level + 1)
            self.gpu.dispatch(
                command, "scanAdd", count, {"n": count, "dataBase": base, "prefixBase": 0},
                data=dst, prefix=prefixes,
            )

    def allocate_pairs(self, count):
        if count <= self.pair_capacity:
            return
        blocks = ceil_div(count, SCAN)
        self.key_a = self.gpu.buffer(count * 8, "Tile/depth keys A")
        self.key_b = self.gpu.buffer(count * 8, "Tile/depth keys B")
        self.id_a = self.gpu.buffer(count * 4, "Splat IDs A")
        self.id_b = self.gpu.buffer(count * 4, "Splat IDs B")
        self.hist = self.gpu.buffer(blocks * 16 * 4, "Radix histogram")
        self.hist_ex = self.gpu.buffer(blocks * 16 * 4, "Scanned radix histogram")
        self.pair_capacity = count

    def sort_pairs(self, count):
        gpu = self.gpu
        keys, keys_out = self.key_a, self.key_b
        ids, ids_out = self.id_a, self.id_b
        blocks = ceil_div(count, SCAN)
        # Higher tile digits are all zero, so their stable passes are identities.
        bits = 32 + (self.tiles - 1).bit_length()
        for shift in range(0, bits, 4):
            pc = {"n": count, "blocks": blocks, "shift": shift}
            command = gpu.encoder()
            command.clear_buffer(self.totals)
            gpu.dispatch(command, "radixHist", count, pc, keys=keys, hist=self.hist, digitTotals=self.totals)
            for digit in range(16):
                self.scan(command, self.hist, self.hist_ex, blocks, digit * blocks)
            gpu.submit(command)
            totals = read_buffer(self.totals, np.int32, (16,))
            if int(totals.sum()) != count:
                raise RuntimeError(f"Radix histogram totals {totals.sum()} for {count} tile pairs")
            bases = np.empty(16, dtype=np.int32)
            bases[0] = 0
            np.cumsum(totals[:-1], out=bases[1:])
            command = gpu.encoder()
            command.upload_buffer_data(self.bases, 0, bases)
            gpu.dispatch(
                command, "radixScatter", count, pc, keysIn=keys, idsIn=ids, keysOut=keys_out,
                idsOut=ids_out, histEx=self.hist_ex, digitBase=self.bases,
            )
            gpu.submit(command)
            keys, keys_out = keys_out, keys
            ids, ids_out = ids_out, ids
        return keys, ids

    def render(self, gauss, count, camera, with_depth):
        gpu = self.gpu
        command = gpu.encoder()
        gpu.dispatch(command, "project", count, camera, gaussians=gauss, splats=self.splats)
        tile = self.tile | {"count": count}
        gpu.dispatch(command, "countTiles", count, tile, splats=self.splats, counts=self.counts)
        self.scan(command, self.counts, self.offsets, count)
        command.copy_buffer(self.tail, 0, self.offsets, (count - 1) * 4, 4)
        command.copy_buffer(self.tail, 4, self.counts, (count - 1) * 4, 4)
        gpu.submit(command)
        offset, last = map(int, read_buffer(self.tail, np.int32, (2,)))
        self.pair_count = offset + last
        if offset < 0 or last < 0 or self.pair_count > 2_000_000_000:
            raise RuntimeError("Tile pair count exceeds the rasterizer limit")
        if self.pair_count == 0:
            command = gpu.encoder()
            command.clear_buffer(self.image)
            command.clear_buffer(self.blend)
            gpu.submit(command)
            return
        self.allocate_pairs(self.pair_count)
        command = gpu.encoder()
        gpu.dispatch(
            command, "writeKeys", count, tile,
            splats=self.splats, offsets=self.offsets, keys=self.key_a, ids=self.id_a,
        )
        gpu.submit(command)
        keys, self.ids = self.sort_pairs(self.pair_count)
        command = gpu.encoder()
        command.clear_buffer(self.ranges)
        gpu.dispatch(
            command, "markRanges", self.pair_count, {"n": self.pair_count, "tiles": self.tiles},
            keys=keys, ranges=self.ranges,
        )
        gpu.dispatch(
            command, "blendRgbEd" if with_depth else "blendRgb",
            (self.tiles_x * 128, self.tiles_y, 1), self.tile,
            splats=self.splats, ranges=self.ranges, ids=self.ids, image=self.image, pixels=self.blend,
        )
        gpu.submit(command)

    def backward(self, command, accum, image_grad, depth_grad, depth_weight, with_depth):
        if self.pair_count == 0:
            return
        buffers = {
            "splats": self.splats, "ranges": self.ranges, "ids": self.ids,
            "imageGrad": image_grad, "accum": accum, "pixels": self.blend,
        }
        if with_depth:
            buffers["depthGrad"] = depth_grad
        self.gpu.dispatch(
            command, "blendBackwardRgbEd" if with_depth else "blendBackwardRgb",
            (self.tiles_x * 128, self.tiles_y, 1), self.tile | {"depthWeight": depth_weight},
            **buffers,
        )
