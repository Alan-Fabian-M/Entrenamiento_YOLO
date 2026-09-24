"""Walks a CubiCasa5K `model.svg` tree and extracts annotated polygons in
absolute image-pixel space, resolving cumulative nested `transform` attributes.
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Set, Tuple, Union
from lxml import etree
import numpy as np

from training.cubicasa.svg_transform import SvgTransform

logger = logging.getLogger(__name__)

_POINTS_PAIR_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
# Only straight-line path commands (M/L/H/V/Z) are supported; icons in CubiCasa5K
# are drawn with straight vertices. Curved commands (C/S/Q/T/A) belong to walls
# or decorative elements outside this pipeline's scope.
_UNSUPPORTED_PATH_COMMANDS = set("CSQTAcsqta")
_PATH_COMMAND_RE = re.compile(r"([MLHVZmlhvz])([^MLHVZmlhvz]*)")


class SvgParsingError(Exception):
    """Raised when an SVG file or one of its elements cannot be parsed."""
    pass


@dataclass
class RawSvgAnnotation:
    raw_class: str
    points_px: np.ndarray
    element_tag: str


class CubiCasaSvgParser:
    """Extracts class-annotated polygons from a CubiCasa5K SVG in image-pixel space."""

    @staticmethod
    def load_svg_root(svg_path: Union[str, Path]) -> etree._Element:
        svg_path = Path(svg_path)
        if not svg_path.exists():
            raise SvgParsingError(f"SVG file not found: {svg_path}")
        try:
            tree = etree.parse(str(svg_path))
        except etree.XMLSyntaxError as exc:
            raise SvgParsingError(f"Malformed SVG at {svg_path}: {exc}") from exc
        return tree.getroot()

    @staticmethod
    def _local_tag(element: etree._Element) -> str:
        tag = element.tag
        if isinstance(tag, str) and "}" in tag:
            return tag.split("}", 1)[1]
        return tag

    @staticmethod
    def get_viewbox_scale(
        svg_root: etree._Element, image_shape: Tuple[int, int]
    ) -> Tuple[float, float]:
        """Return (scale_x, scale_y) to map SVG viewBox coordinates to the
        raster image's pixel space.
        """
        img_h, img_w = image_shape[0], image_shape[1]

        viewbox = svg_root.get("viewBox")
        if viewbox:
            parts = [float(p) for p in _POINTS_PAIR_RE.findall(viewbox)]
            if len(parts) == 4:
                _, _, vb_w, vb_h = parts
                if vb_w > 0 and vb_h > 0:
                    return img_w / vb_w, img_h / vb_h

        width_attr = svg_root.get("width")
        height_attr = svg_root.get("height")
        if width_attr and height_attr:
            try:
                svg_w = float(_POINTS_PAIR_RE.findall(width_attr)[0])
                svg_h = float(_POINTS_PAIR_RE.findall(height_attr)[0])
                if svg_w > 0 and svg_h > 0:
                    return img_w / svg_w, img_h / svg_h
            except (IndexError, ValueError):
                pass

        return 1.0, 1.0

    @staticmethod
    def _get_class_attr(element: etree._Element) -> Optional[str]:
        return element.get("class")

    @staticmethod
    def _extract_points(element: etree._Element) -> Optional[np.ndarray]:
        tag = CubiCasaSvgParser._local_tag(element)

        if tag == "polygon" or tag == "polyline":
            raw = element.get("points")
            if not raw:
                return None
            nums = [float(n) for n in _POINTS_PAIR_RE.findall(raw)]
            if len(nums) < 6 or len(nums) % 2 != 0:
                return None
            return np.array(nums, dtype=np.float64).reshape(-1, 2)

        if tag == "rect":
            try:
                x = float(element.get("x", "0"))
                y = float(element.get("y", "0"))
                w = float(element.get("width", "0"))
                h = float(element.get("height", "0"))
            except ValueError:
                return None
            if w <= 0 or h <= 0:
                return None
            return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float64)

        if tag == "path":
            d = element.get("d")
            if not d:
                return None
            if any(cmd in d for cmd in _UNSUPPORTED_PATH_COMMANDS):
                logger.warning("Skipping <path> with unsupported curve commands: %r", d[:60])
                return None
            return CubiCasaSvgParser._parse_straight_path(d)

        return None

    @staticmethod
    def _parse_straight_path(d: str) -> Optional[np.ndarray]:
        points: List[Tuple[float, float]] = []
        cur = (0.0, 0.0)
        for cmd, arg_str in _PATH_COMMAND_RE.findall(d):
            nums = [float(n) for n in _POINTS_PAIR_RE.findall(arg_str)]
            is_relative = cmd.islower()
            cmd_upper = cmd.upper()

            if cmd_upper == "M" or cmd_upper == "L":
                for i in range(0, len(nums) - 1, 2):
                    x, y = nums[i], nums[i + 1]
                    if is_relative:
                        x, y = cur[0] + x, cur[1] + y
                    cur = (x, y)
                    points.append(cur)
            elif cmd_upper == "H":
                for x in nums:
                    x = cur[0] + x if is_relative else x
                    cur = (x, cur[1])
                    points.append(cur)
            elif cmd_upper == "V":
                for y in nums:
                    y = cur[1] + y if is_relative else y
                    cur = (cur[0], y)
                    points.append(cur)
            elif cmd_upper == "Z":
                continue

        if len(points) < 3:
            return None
        return np.array(points, dtype=np.float64)

    @classmethod
    def walk_and_collect(
        cls,
        svg_root: etree._Element,
        image_shape: Tuple[int, int],
        target_classes: Optional[Set[str]] = None,
    ) -> List[RawSvgAnnotation]:
        """Recursively walk the SVG tree accumulating transforms, and return
        every annotated polygon/path/rect element found. If `target_classes`
        is None, all classed elements are collected (useful for auditing).
        """
        scale_x, scale_y = cls.get_viewbox_scale(svg_root, image_shape)
        scale_matrix = np.array(
            [[scale_x, 0, 0], [0, scale_y, 0], [0, 0, 1]], dtype=np.float64
        )

        results: List[RawSvgAnnotation] = []

        def _walk(element: etree._Element, accumulated: np.ndarray) -> None:
            transform_attr = element.get("transform")
            local_matrix = SvgTransform.parse_transform_attr(transform_attr)
            current = SvgTransform.compose(accumulated, local_matrix)

            raw_class = cls._get_class_attr(element)
            if raw_class:
                normalized = raw_class.strip()
                if target_classes is None or normalized.lower() in target_classes:
                    points = cls._extract_points(element)
                    if points is not None:
                        world_points = SvgTransform.apply_to_points(current, points)
                        px_points = SvgTransform.apply_to_points(scale_matrix, world_points)
                        results.append(
                            RawSvgAnnotation(
                                raw_class=normalized,
                                points_px=px_points,
                                element_tag=cls._local_tag(element),
                            )
                        )

            for child in element:
                if isinstance(child.tag, str):
                    _walk(child, current)

        _walk(svg_root, SvgTransform.IDENTITY.copy())
        return results
