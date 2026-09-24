"""CLI entrypoint for the CubiCasa5K -> YOLOv8-OBB dataset conversion pipeline.

Usage:
    python -m training.cli build --cubicasa-root <path> --output-root <path> \\
        --train-split <path> --val-split <path> --test-split <path> [--visualize]

    python -m training.cli audit-classes --cubicasa-root <path> \\
        --train-split <path> --val-split <path> --test-split <path>
"""

import argparse
import logging
from pathlib import Path

import cv2

from training.cubicasa.class_mapping import ClassMapper
from training.cubicasa.dataset_builder import BuildConfig, YoloObbDatasetBuilder
from training.cubicasa.svg_parser import CubiCasaSvgParser

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CubiCasa5K -> YOLOv8-OBB dataset tools")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def _add_common_args(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--cubicasa-root", type=Path, required=True)
        sub.add_argument("--train-split", type=Path, required=True)
        sub.add_argument("--val-split", type=Path, required=True)
        sub.add_argument("--test-split", type=Path, required=True)
        sub.add_argument("--image-filename", type=str, default="F1_scaled.png")
        sub.add_argument("--svg-filename", type=str, default="model.svg")

    build_parser = subparsers.add_parser("build", help="Build the full YOLO-OBB dataset")
    _add_common_args(build_parser)
    build_parser.add_argument("--output-root", type=Path, required=True)
    build_parser.add_argument("--min-box-dim-px", type=float, default=3.0)
    build_parser.add_argument("--visualize", action="store_true")
    build_parser.add_argument("--vis-sample-rate", type=float, default=0.02)

    audit_parser = subparsers.add_parser(
        "audit-classes", help="Report unmapped raw SVG classes without writing a dataset"
    )
    _add_common_args(audit_parser)
    audit_parser.add_argument("--output", type=Path, default=Path("unmapped_classes.txt"))

    return parser


def _splits_from_args(args: argparse.Namespace) -> dict:
    return {"train": args.train_split, "val": args.val_split, "test": args.test_split}


def cmd_build(args: argparse.Namespace) -> None:
    config = BuildConfig(
        cubicasa_root=args.cubicasa_root,
        output_root=args.output_root,
        splits=_splits_from_args(args),
        image_filename=args.image_filename,
        svg_filename=args.svg_filename,
        min_box_dim_px=args.min_box_dim_px,
        generate_visualizations=args.visualize,
        vis_sample_rate=args.vis_sample_rate,
    )
    report = YoloObbDatasetBuilder.build(config)

    logger.info("Total samples: %d", report.total_samples)
    logger.info("Skipped samples: %d", report.skipped_samples)
    logger.info("Instances per class: %s", report.instances_per_class)
    if report.unmapped_raw_classes:
        logger.warning("Unmapped raw classes seen: %s", sorted(report.unmapped_raw_classes))


def cmd_audit_classes(args: argparse.Namespace) -> None:
    splits = _splits_from_args(args)
    unrecognized = set()

    for split, split_path in splits.items():
        relative_paths = YoloObbDatasetBuilder.read_split_file(split_path)
        for relative_path in relative_paths:
            sample_dir = args.cubicasa_root / relative_path
            image_path = sample_dir / args.image_filename
            svg_path = sample_dir / args.svg_filename
            if not image_path.exists() or not svg_path.exists():
                continue

            image = cv2.imread(str(image_path))
            if image is None:
                continue

            try:
                svg_root = CubiCasaSvgParser.load_svg_root(svg_path)
                annotations = CubiCasaSvgParser.walk_and_collect(svg_root, image.shape)
            except Exception:
                logger.exception("Failed to parse %s", svg_path)
                continue

            for ann in annotations:
                ClassMapper.map_to_unified(ann.raw_class)

    unrecognized = ClassMapper.get_unrecognized_classes()
    args.output.write_text("\n".join(sorted(unrecognized)) + ("\n" if unrecognized else ""))
    logger.info("Found %d unrecognized raw classes. Written to %s", len(unrecognized), args.output)


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.command == "build":
        cmd_build(args)
    elif args.command == "audit-classes":
        cmd_audit_classes(args)


if __name__ == "__main__":
    main()
