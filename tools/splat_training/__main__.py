"""Run the Python training port on a Pocket Scanner recording."""

import argparse
import json
import logging
import math
import random
import sys
import time
from dataclasses import asdict, fields, replace
from itertools import islice
from pathlib import Path
from zipfile import BadZipFile, ZipFile

from PIL import Image

from .capture import Capture
from .frame_selection import (
    NEAR_POSITION_M,
    NEAR_ROTATION_DEGREES,
    select_training_frames,
)
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
    parser.add_argument(
        "--steps-per-epoch", type=at_least(1),
        help="run a fixed number of Gaussian updates per epoch for comparisons across datasets",
    )
    parser.add_argument(
        "--coarse-epochs", type=at_least(0), default=0,
        help="train at half resolution for the first N epochs and evaluate at full resolution throughout",
    )
    parser.add_argument("--shuffle-seed", type=int, help="deterministically shuffle each traversal of train views")
    parser.add_argument("--budget", type=at_least(3), default=3_000_000)
    parser.add_argument("--validation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--validation-percent", choices=(5, 8, 10), type=int, default=5)
    parser.add_argument("--validation-every", type=at_least(1), default=5)
    parser.add_argument(
        "--filter-redundant-frames", action=argparse.BooleanOptionalAction, default=True,
        help="discard train views whose measured coverage and parallax are already represented (default: enabled)",
    )
    parser.add_argument(
        "--filter-pose-distance", type=float, default=NEAR_POSITION_M, metavar="METERS",
        help="maximum distance from an original pose to a retained representative",
    )
    parser.add_argument(
        "--filter-pose-angle", type=float, default=NEAR_ROTATION_DEGREES, metavar="DEGREES",
        help="maximum relative camera rotation to a retained representative",
    )
    parser.add_argument("--max-frames", type=at_least(1), help="debug with the first N usable views by ID")
    parser.add_argument("--hyper", type=Path, help="JSON object with overrides named as in SplatHyper")
    parser.add_argument("--tensorboard", action="store_true", help="write per-view and per-epoch scalars")
    parser.add_argument("--save-validation-scenes", action="store_true", help="export a PLY after each validation pass")
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
    if not 0 < hyper.lrColorEnd <= 1:
        raise ValueError("lrColorEnd must be > 0 and <= 1")
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
                if value is not None:
                    summary.value.add(tag=f"{prefix}/{name}", simple_value=value)
            self.board.add_event(self.event_type(wall_time=time.time(), step=step, summary=summary))

    def epoch(self, epoch, split, metrics, count):
        row = {"epoch": epoch, "split": split, **metrics, "gaussians": count}
        with (self.output / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        self.scalars(f"epoch/{split}", epoch, metrics | {"gaussians": count})
        if self.board is not None:
            self.board.flush()
        raw = f"  raw PSNR {metrics['raw_psnr']:.3f}" if split == "val" else ""
        logger.info(
            "Epoch %d %s  L1 %.5f  SSIM %.5f  PSNR %.3f%s  depth %.5f  Gaussians %d",
            epoch, split, metrics["l1"], metrics["ssim"], metrics["psnr"], raw, metrics["depth"], count,
        )

    def close(self):
        if self.board is not None:
            self.board.close()


def run_pass(capture, views, trainer, logs, split, epoch, epochs):
    train = split == "train"
    size = (trainer.raster.width, trainer.raster.height)
    sums = dict.fromkeys(Metrics.__dataclass_fields__, 0.0)
    if not train:
        sums.update(dict.fromkeys(("raw_l1", "raw_ssim", "raw_psnr"), 0.0))
    depth_targets = depth_matches = 0
    depth_abs_sum = depth_square_sum = 0.0
    start = time.monotonic()
    for number, (index, view) in enumerate(views, start=1):
        try:
            rgb, depth = capture.images(view, size)
            metrics = asdict(trainer.step(view, index, rgb, depth, train=train))
            if not train:
                if trainer.uses_grid:
                    metrics.update(trainer.raw_color_metrics())
                else:
                    metrics.update(raw_l1=metrics["l1"], raw_ssim=metrics["ssim"], raw_psnr=metrics["psnr"])
                targets, matches, abs_sum, square_sum = trainer.validation_depth_totals(depth)
                depth_targets += targets
                depth_matches += matches
                depth_abs_sum += abs_sum
                depth_square_sum += square_sum
        except (ValueError, RuntimeError, OSError, KeyError) as error:
            raise RuntimeError(f"Epoch {epoch}, {split}, frame {view.id}: {error}") from error
        for key, value in metrics.items():
            sums[key] += value
        logs.scalars(
            f"view/{split}", trainer.grid_step if trainer.uses_grid else trainer.gauss_step,
            metrics | {
                "frame_id": view.id, "gaussians": trainer.count, "tile_pairs": trainer.raster.pair_count,
                "lr_grid": trainer.hyper.grid_rate(trainer.grid_step, trainer.grid_steps) if trainer.uses_grid else 0.0,
                "lr_mean": trainer.hyper.mean_rate(trainer.gauss_step, trainer.mean_steps) if train else 0.0,
                "lr_color": trainer.hyper.color_rate(trainer.gauss_step, trainer.mean_steps) if train else 0.0,
            },
        )
        if not train:
            val = logs.output / "val"
            val.mkdir(exist_ok=True)
            Image.fromarray(trainer.corrected_rgb()).save(val / f"{view.id}.png")
        if number == 1 or number % 25 == 0 or number == len(views):
            logger.info("Epoch %d/%d %s %d/%d (frame %d)", epoch, epochs, split, number, len(views), view.id)
    averaged = {key: value / len(views) for key, value in sums.items()}
    if not train:
        averaged.update({
            "depth_coverage": depth_matches / depth_targets if depth_targets else None,
            "depth_mae_m": depth_abs_sum / depth_matches if depth_matches else None,
            "depth_rmse_m": math.sqrt(depth_square_sum / depth_matches) if depth_matches else None,
            "depth_target_pixels": depth_targets,
            "depth_matched_pixels": depth_matches,
        })
    logs.epoch(epoch, split, averaged | {
        "gauss_step": trainer.gauss_step, "grid_step": trainer.grid_step,
        "width": size[0], "height": size[1],
        "seconds": time.monotonic() - start,
    }, trainer.count)


def training_views(views, shuffle_seed):
    traversal = list(views)
    rng = random.Random(shuffle_seed) if shuffle_seed is not None else None
    while traversal:
        if rng is not None:
            rng.shuffle(traversal)
        yield from traversal


def write_json(path, values):
    path.write_text(json.dumps(values, indent=2, allow_nan=False) + "\n")


def run(args):
    hyper = read_hyper(args.hyper)
    if args.coarse_epochs >= args.epochs:
        raise ValueError("--coarse-epochs must leave at least one full-resolution epoch")
    if not math.isfinite(args.filter_pose_distance) or args.filter_pose_distance <= 0:
        raise ValueError("--filter-pose-distance must be finite and positive")
    if not 0 < args.filter_pose_angle <= 180:
        raise ValueError("--filter-pose-angle must be > 0 and <= 180 degrees")
    with ZipFile(args.scan) as archive:
        capture = Capture(archive)
        views = list(enumerate(capture.views[:args.max_frames]))
        step = math.floor(100 / args.validation_percent + 0.5)
        val = [(i, view) for i, view in views if args.validation and len(views) > 1 and i % step == 0]
        val_indices = {i for i, _ in val}
        train = [(i, view) for i, view in views if i not in val_indices]
        selection_info = {"enabled": args.filter_redundant_frames, "sourceTrainCount": len(train)}
        mesh = capture.training_mesh(
            args.budget, args.mesh, include_coverage=args.filter_redundant_frames and bool(train),
        )
        if mesh.coverage_points is not None:
            logger.info("Checking measured coverage and parallax for %d train views", len(train))
            selection = select_training_frames(
                capture, [view for _, view in train], mesh.coverage_points, mesh.voxel,
                near_position=args.filter_pose_distance, near_rotation=args.filter_pose_angle,
            )
            selection_info.update({
                "surfaceSamples": len(mesh.coverage_points),
                "coveredSamples": selection.covered_samples,
                "parallaxSamples": selection.parallax_samples,
                "retainedCoveredSamples": selection.retained_covered_samples,
                "retainedParallaxSamples": selection.retained_parallax_samples,
                "nearPositionMeters": args.filter_pose_distance,
                "nearRotationDegrees": args.filter_pose_angle,
                "droppedTrainIds": [train[i][1].id for i in selection.dropped_indices],
                "unverifiedTrainIds": [train[i][1].id for i in selection.unverified_indices],
            })
            train = [train[i] for i in selection.kept_indices]
            logger.info(
                "Kept %d/%d train views, preserving coverage at %d/%d samples and parallax at %d/%d",
                len(train), selection_info["sourceTrainCount"], selection.retained_covered_samples,
                selection.covered_samples, selection.retained_parallax_samples, selection.parallax_samples,
            )
        # Allocate grids only for the views used by the run, in source order.
        grid_indices = {index: i for i, index in enumerate(sorted(index for index, _ in train + val))}
        train = [(grid_indices[index], view) for index, view in train]
        val = [(grid_indices[index], view) for index, view in val]
        if not train:
            raise ValueError("No train views. Disable validation or use more frames.")
        steps_per_epoch = args.steps_per_epoch or len(train)
        mean_steps = args.epochs * steps_per_epoch
        grid_steps = mean_steps + (args.epochs // args.validation_every) * len(val) if hyper.lrGrid else 0
        voxel = mesh.voxel
        gauss, extent = seed_gaussians(mesh.seed, voxel, hyper)
        del mesh
        logger.info(
            "%s: %d train, %d validation, %d×%d, %d seed Gaussians, voxel %.3f m",
            args.scan.name, len(train), len(val), capture.width, capture.height, len(gauss), voxel,
        )
        gpu = Gpu(args.backend)
        full_size = (capture.width, capture.height)
        coarse_size = tuple(max(16, side // 2) for side in full_size)
        trainer = Trainer(
            gpu, gauss, extent, *(coarse_size if args.coarse_epochs else full_size), len(grid_indices),
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
            "stepsPerEpoch": steps_per_epoch, "shuffleSeed": args.shuffle_seed,
            "coarseEpochs": args.coarse_epochs, "coarseSize": coarse_size if args.coarse_epochs else None,
            "meanSteps": mean_steps, "gridSteps": grid_steps,
            "trainIds": [view.id for _, view in train], "valIds": [view.id for _, view in val],
            "frameSelection": selection_info,
        })
        write_json(args.output / "hyperparameters.json", asdict(hyper) | {
            "budget": args.budget, "extent": extent, "meanSteps": mean_steps, "gridSteps": grid_steps,
        })
        logs = Logs(args.output, args.tensorboard)
        try:
            stream = training_views(train, args.shuffle_seed)
            for epoch in range(1, args.epochs + 1):
                trainer.set_resolution(*(coarse_size if epoch <= args.coarse_epochs else full_size))
                run_pass(capture, list(islice(stream, steps_per_epoch)), trainer, logs, "train", epoch, args.epochs)
                if val and epoch % args.validation_every == 0:
                    trainer.set_resolution(*full_size)
                    run_pass(capture, val, trainer, logs, "val", epoch, args.epochs)
                    if args.save_validation_scenes:
                        trainer.export(args.output / f"scene-{trainer.gauss_step:06}.ply")
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
