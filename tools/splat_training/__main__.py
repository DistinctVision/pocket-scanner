"""Run the Python training port on a Pocket Scanner recording."""

import argparse
import json
import logging
import math
import sys
import time
from dataclasses import asdict, fields, replace
from pathlib import Path
from zipfile import BadZipFile, ZipFile

from PIL import Image

from .capture import Capture
from .frame_selection import select_training_frames
from .gpu import Gpu
from .population import Hyper, seed_gaussians
from .trainer import Metrics, Trainer

logger = logging.getLogger(__name__)


def at_least(minimum):
    def parse(value):
        integer = int(value)
        if integer < minimum:
            raise argparse.ArgumentTypeError(f"expected an integer >= {minimum}")
        return integer
    return parse


def options():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scan", type=Path, help="Pocket Scanner .pscan archive")
    parser.add_argument("--output", type=Path, required=True, help="new directory for training outputs")
    parser.add_argument("--mesh", type=Path, help="override the mesh.ply inside the recording")
    parser.add_argument("--backend", choices=("auto", "metal", "vulkan"), default="auto")
    parser.add_argument("--epochs", type=at_least(1), default=100)
    parser.add_argument("--budget", type=at_least(3), default=3_000_000)
    parser.add_argument("--validation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--validation-percent", choices=(5, 8, 10), type=int, default=5)
    parser.add_argument("--validation-every", type=at_least(1), default=5)
    parser.add_argument(
        "--filter-redundant-frames", action=argparse.BooleanOptionalAction, default=True,
        help="discard train views whose measured coverage and parallax are already represented (default: enabled)",
    )
    parser.add_argument("--max-frames", type=at_least(1), help="debug with the first N usable views by ID")
    parser.add_argument("--hyper", type=Path, help="JSON object with overrides named as in SplatHyper")
    parser.add_argument("--tensorboard", action="store_true", help="write per-view and per-epoch scalars")
    return parser.parse_args()


def read_hyper(path):
    hyper = Hyper()
    if path is None:
        return hyper
    overrides = json.loads(path.read_text())
    if not isinstance(overrides, dict):
        raise TypeError("--hyper expects a JSON object")
    types = {field.name: field.type for field in fields(Hyper)}
    for name, value in overrides.items():
        if name not in types:
            raise ValueError(f"Unknown hyperparameter {name}")
        if type(value) not in (int, float) or not math.isfinite(value) or (
            types[name] is int and type(value) is not int
        ):
            raise ValueError(f"Invalid value for hyperparameter {name}")
    hyper = replace(hyper, **overrides)
    if not 1 / 255 < hyper.seedOpacity <= 1:
        raise ValueError("seedOpacity must be > 1/255 and <= 1")
    if not (0 <= hyper.beta1 < 1 and 0 <= hyper.beta2 < 1 and hyper.adamEps > 0):
        raise ValueError("Adam requires beta1 and beta2 in [0, 1) and positive adamEps")
    return hyper


class Logs:
    def __init__(self, output, tensorboard):
        self.output = output
        self.board = None
        if tensorboard:
            try:
                from tensorboard.compat.proto.event_pb2 import Event
                from tensorboard.compat.proto.summary_pb2 import Summary
                from tensorboard.summary.writer.event_file_writer import EventFileWriter
            except ImportError as error:
                raise ValueError("Install tensorboard to use --tensorboard") from error
            self.event_type, self.summary_type = Event, Summary
            self.board = EventFileWriter(str(output / "tensorboard"))

    def scalars(self, prefix, step, values):
        if self.board is not None:
            summary = self.summary_type()
            for name, value in values.items():
                summary.value.add(tag=f"{prefix}/{name}", simple_value=value)
            self.board.add_event(self.event_type(wall_time=time.time(), step=step, summary=summary))

    def epoch(self, epoch, split, metrics, count):
        row = {"epoch": epoch, "split": split, **metrics, "gaussians": count}
        with (self.output / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        self.scalars(f"epoch/{split}", epoch, metrics | {"gaussians": count})
        if self.board is not None:
            self.board.flush()
        logger.info(
            "Epoch %d %s  L1 %.5f  SSIM %.5f  PSNR %.3f  depth %.5f  Gaussians %d",
            epoch, split, metrics["l1"], metrics["ssim"], metrics["psnr"], metrics["depth"], count,
        )

    def close(self):
        if self.board is not None:
            self.board.close()


def run_pass(capture, views, trainer, logs, split, epoch, epochs):
    train = split == "train"
    sums = dict.fromkeys(Metrics.__dataclass_fields__, 0.0)
    for number, (index, view) in enumerate(views, start=1):
        try:
            rgb, depth = capture.images(view)
            metrics = asdict(trainer.step(view, index, rgb, depth, train=train))
        except (ValueError, RuntimeError, OSError, KeyError) as error:
            raise RuntimeError(f"Epoch {epoch}, {split}, frame {view.id}: {error}") from error
        for key, value in metrics.items():
            sums[key] += value
        logs.scalars(
            f"view/{split}", trainer.grid_step,
            metrics | {
                "frame_id": view.id, "gaussians": trainer.count, "tile_pairs": trainer.raster.pair_count,
                "lr_grid": trainer.hyper.grid_rate(trainer.grid_step, trainer.grid_steps),
                "lr_mean": trainer.hyper.mean_rate(trainer.gauss_step, trainer.mean_steps) if train else 0.0,
            },
        )
        if not train:
            val = logs.output / "val"
            val.mkdir(exist_ok=True)
            Image.fromarray(trainer.corrected_rgb()).save(val / f"{view.id}.png")
        if number == 1 or number % 25 == 0 or number == len(views):
            logger.info("Epoch %d/%d %s %d/%d (frame %d)", epoch, epochs, split, number, len(views), view.id)
    logs.epoch(epoch, split, {key: value / len(views) for key, value in sums.items()}, trainer.count)


def write_json(path, values):
    path.write_text(json.dumps(values, indent=2, allow_nan=False) + "\n")


def run(args):
    hyper = read_hyper(args.hyper)
    with ZipFile(args.scan) as archive:
        capture = Capture(archive)
        views = list(enumerate(capture.views[:args.max_frames]))
        step = math.floor(100 / args.validation_percent + 0.5) if args.validation else len(views) + 1
        train = [(i, view) for i, view in views if not args.validation or i % step]
        val = [(i, view) for i, view in views if args.validation and i % step == 0]
        selection_info = {"enabled": args.filter_redundant_frames, "sourceTrainCount": len(train)}
        mesh = capture.training_mesh(
            args.budget, args.mesh, include_coverage=args.filter_redundant_frames and bool(train),
        )
        if mesh.coverage_points is not None:
            logger.info("Checking measured coverage and parallax for %d train views", len(train))
            selection = select_training_frames(capture, [view for _, view in train], mesh.coverage_points)
            selection_info.update({
                "surfaceSamples": len(mesh.coverage_points),
                "coveredSamples": selection.covered_samples,
                "parallaxSamples": selection.parallax_samples,
                "droppedTrainIds": [train[i][1].id for i in selection.dropped_indices],
                "unverifiedTrainIds": [train[i][1].id for i in selection.unverified_indices],
            })
            train = [train[i] for i in selection.kept_indices]
            logger.info(
                "Kept %d/%d train views, with measured coverage at %d mesh samples and parallax at %d",
                len(train), selection_info["sourceTrainCount"], selection.covered_samples,
                selection.parallax_samples,
            )
        # Allocate grids only for the views used by the run, in source order.
        grid_indices = {index: i for i, index in enumerate(sorted(index for index, _ in train + val))}
        train = [(grid_indices[index], view) for index, view in train]
        val = [(grid_indices[index], view) for index, view in val]
        mean_steps = max(1, args.epochs * len(train))
        grid_steps = mean_steps + (args.epochs // args.validation_every) * len(val)
        voxel = mesh.voxel
        gauss, extent = seed_gaussians(mesh.seed, voxel, hyper)
        del mesh
        logger.info(
            "%s: %d train, %d validation, %d×%d, %d seed Gaussians, voxel %.3f m",
            args.scan.name, len(train), len(val), capture.width, capture.height, len(gauss), voxel,
        )
        gpu = Gpu(args.backend)
        trainer = Trainer(
            gpu, gauss, extent, capture.width, capture.height, len(grid_indices),
            hyper, args.budget, mean_steps, grid_steps,
        )
        del gauss
        args.output.mkdir(parents=True)
        write_json(args.output / "run.json", {
            "scan": args.scan.name, "mesh": args.mesh.name if args.mesh else "mesh.ply",
            "backend": gpu.backend, "adapter": gpu.device.info.adapter_name,
            "width": capture.width, "height": capture.height, "voxel": voxel,
            "epochs": args.epochs, "validation": args.validation, "percent": args.validation_percent,
            "every": args.validation_every, "budget": args.budget, "seedCount": trainer.count,
            "meanSteps": mean_steps, "gridSteps": grid_steps,
            "trainIds": [view.id for _, view in train], "valIds": [view.id for _, view in val],
            "frameSelection": selection_info,
        })
        write_json(args.output / "hyperparameters.json", asdict(hyper) | {
            "budget": args.budget, "extent": extent, "meanSteps": mean_steps, "gridSteps": grid_steps,
        })
        logs = Logs(args.output, args.tensorboard)
        try:
            for epoch in range(1, args.epochs + 1):
                if train:
                    run_pass(capture, train, trainer, logs, "train", epoch, args.epochs)
                if val and epoch % args.validation_every == 0:
                    run_pass(capture, val, trainer, logs, "val", epoch, args.epochs)
            trainer.export(args.output / "scene.ply")
        finally:
            logs.close()


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = options()
    if args.output.exists():
        logger.error("Output directory already exists: %s", args.output)
        return 1
    try:
        run(args)
    except (BadZipFile, OSError, TypeError, ValueError, RuntimeError, KeyError) as error:
        logger.error("%s", error)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
