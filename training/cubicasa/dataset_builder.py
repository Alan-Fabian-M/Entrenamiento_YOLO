"""Orchestrates the full CubiCasa5K -> YOLOv8-OBB dataset build."""

import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
from tqdm import tqdm

from training.cubicasa.class_mapping import ClassMapper, YOLO_CLASSES
from training.cubicasa.obb_converter import ObbConverter
from training.cubicasa.svg_parser import CubiCasaSvgParser
from training.cubicasa.visualizer import Visualizer

logger = logging.getLogger(__name__)


class DatasetBuilderError(Exception):
    """Raised when the dataset build process cannot proceed."""
    pass


@dataclass
class BuildConfig:
    cubicasa_root: Path
    output_root: Path
    splits: Dict[str, Path]  # e.g. {"train": train.txt, "val": val.txt, "test": test.txt}
    image_filename: str = "F1_scaled.png"
    svg_filename: str = "model.svg"
    min_box_dim_px: float = 3.0
    generate_visualizations: bool = False
    vis_sample_rate: float = 0.02


@dataclass
class BuildReport:
    total_samples: int = 0
    skipped_samples: int = 0
    instances_per_class: Dict[str, int] = field(default_factory=dict)
    unmapped_raw_classes: set = field(default_factory=set)


class YoloObbDatasetBuilder:
    """Builds a YOLOv8-OBB-ready dataset (images/, labels/, data.yaml) from
    CubiCasa5K SVG annotations.
    """

    @staticmethod
    def read_split_file(split_path: Path) -> List[str]:
        if not split_path.exists():
            raise DatasetBuilderError(f"Split file not found: {split_path}")
        lines = split_path.read_text().splitlines()
        return [line.strip().lstrip("/") for line in lines if line.strip()]

    @staticmethod
    def _sample_id(relative_path: str) -> str:
        return relative_path.strip("/").replace("/", "_").replace("\\", "_")

    @staticmethod
    def process_sample(
        sample_dir: Path,
        image_filename: str,
        svg_filename: str,
        min_box_dim_px: float,
    ) -> Optional[Tuple[np.ndarray, List[str]]]:
        """Run the full per-sample pipeline: load image, parse SVG, map
        classes, convert to OBB, filter degenerate boxes.
        Returns (image, label_lines) or None if the sample cannot be processed.
        """
        image_path = sample_dir / image_filename
        svg_path = sample_dir / svg_filename

        if not image_path.exists() or not svg_path.exists():
            logger.warning("Skipping sample %s: missing image or SVG", sample_dir)
            return None

        image = cv2.imread(str(image_path))
        if image is None:
            logger.warning("Skipping sample %s: could not decode image", sample_dir)
            return None

        try:
            svg_root = CubiCasaSvgParser.load_svg_root(svg_path)
            annotations = CubiCasaSvgParser.walk_and_collect(svg_root, image.shape)
        except Exception:
            logger.exception("Skipping sample %s: SVG parsing failed", sample_dir)
            return None

        label_lines: List[str] = []
        img_h, img_w = image.shape[0], image.shape[1]

        for ann in annotations:
            unified = ClassMapper.map_to_unified(ann.raw_class)
            if unified is None:
                continue

            corners_px = ObbConverter.polygon_to_obb(ann.points_px)
            if corners_px is None:
                continue

            width = np.linalg.norm(corners_px[1] - corners_px[0])
            height = np.linalg.norm(corners_px[2] - corners_px[1])
            if min(width, height) < min_box_dim_px:
                continue

            class_id = ClassMapper.class_id(unified)
            corners_norm = ObbConverter.normalize_corners(corners_px, img_w, img_h)
            label_lines.append(ObbConverter.to_yolo_obb_line(class_id, corners_norm))

        return image, label_lines

    @staticmethod
    def write_sample_outputs(
        image: np.ndarray,
        label_lines: List[str],
        sample_id: str,
        split: str,
        output_root: Path,
    ) -> None:
        images_dir = output_root / "images" / split
        labels_dir = output_root / "labels" / split
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(images_dir / f"{sample_id}.png"), image)
        (labels_dir / f"{sample_id}.txt").write_text(
            "\n".join(label_lines) + ("\n" if label_lines else "")
        )

    @staticmethod
    def write_data_yaml(output_root: Path, class_names: List[str]) -> None:
        lines = [
            f"path: {output_root}",
            "train: images/train",
            "val: images/val",
            "test: images/test",
            "names:",
        ]
        lines.extend(f"  {i}: {name}" for i, name in enumerate(class_names))
        (output_root / "data.yaml").write_text("\n".join(lines) + "\n")

    @classmethod
    def build(cls, config: BuildConfig) -> BuildReport:
        report = BuildReport()
        vis_dir = config.output_root / "visualizations"
        if config.generate_visualizations:
            vis_dir.mkdir(parents=True, exist_ok=True)

        for split, split_path in config.splits.items():
            relative_paths = cls.read_split_file(split_path)

            for relative_path in tqdm(relative_paths, desc=f"Building {split}"):
                sample_dir = config.cubicasa_root / relative_path
                report.total_samples += 1

                result = cls.process_sample(
                    sample_dir, config.image_filename, config.svg_filename,
                    config.min_box_dim_px,
                )
                if result is None:
                    report.skipped_samples += 1
                    continue

                image, label_lines = result
                sample_id = cls._sample_id(relative_path)
                cls.write_sample_outputs(image, label_lines, sample_id, split, config.output_root)

                for line in label_lines:
                    class_id = int(line.split()[0])
                    class_name = YOLO_CLASSES[class_id]
                    report.instances_per_class[class_name] = (
                        report.instances_per_class.get(class_name, 0) + 1
                    )

                if config.generate_visualizations and random.random() < config.vis_sample_rate:
                    vis_image = Visualizer.draw_obb_labels(image, label_lines, YOLO_CLASSES)
                    cv2.imwrite(str(vis_dir / f"{sample_id}_vis.png"), vis_image)

        report.unmapped_raw_classes = ClassMapper.get_unrecognized_classes()
        cls.write_data_yaml(config.output_root, YOLO_CLASSES)
        return report
