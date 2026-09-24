"""Converts pixel-space polygons into normalized YOLOv8-OBB oriented boxes."""

from dataclasses import dataclass
from typing import Optional
import cv2
import numpy as np
from shapely.geometry import Polygon
from shapely.errors import ShapelyError


class ObbConversionError(Exception):
    """Raised for invalid OBB conversion inputs."""
    pass


@dataclass
class ObbLabel:
    class_id: int
    corners_px: np.ndarray  # (4, 2)


class ObbConverter:
    """Fits an oriented bounding box to a polygon and formats it as a YOLO-OBB label."""

    @staticmethod
    def is_valid_polygon(points_px: np.ndarray, min_area_px: float = 4.0) -> bool:
        """True if `points_px` forms a valid, non-degenerate polygon with area
        at least `min_area_px`.
        """
        if points_px is None or len(points_px) < 3:
            return False
        try:
            polygon = Polygon(points_px)
        except (ShapelyError, ValueError):
            return False
        if not polygon.is_valid or polygon.is_empty:
            return False
        return polygon.area >= min_area_px

    @staticmethod
    def polygon_to_obb(points_px: np.ndarray) -> Optional[np.ndarray]:
        """Fit a minimum-area rotated rectangle to `points_px` and return its
        4 corners in pixel space, or None if the polygon is degenerate.
        """
        if not ObbConverter.is_valid_polygon(points_px):
            return None
        rect = cv2.minAreaRect(points_px.astype(np.float32))
        corners = cv2.boxPoints(rect)
        return ObbConverter.order_corners_clockwise(corners)

    @staticmethod
    def order_corners_clockwise(corners: np.ndarray) -> np.ndarray:
        """Order 4 corners deterministically: clockwise, starting from the
        corner closest to the top-left of their bounding box.
        """
        corners = np.asarray(corners, dtype=np.float64)
        centroid = corners.mean(axis=0)
        angles = np.arctan2(corners[:, 1] - centroid[1], corners[:, 0] - centroid[0])
        order = np.argsort(angles)
        ordered = corners[order]

        start_idx = int(np.argmin(ordered[:, 0] + ordered[:, 1]))
        return np.roll(ordered, -start_idx, axis=0)

    @staticmethod
    def normalize_corners(corners_px: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
        """Normalize pixel corners to [0, 1], clamping slight out-of-canvas overshoot."""
        if img_w <= 0 or img_h <= 0:
            raise ObbConversionError(f"Invalid image dimensions: {img_w}x{img_h}")
        normalized = corners_px.astype(np.float64) / np.array([img_w, img_h], dtype=np.float64)
        return np.clip(normalized, 0.0, 1.0)

    @staticmethod
    def to_yolo_obb_line(class_id: int, corners_norm: np.ndarray) -> str:
        """Format 'class_id x1 y1 x2 y2 x3 y3 x4 y4' with fixed precision."""
        if corners_norm.shape != (4, 2):
            raise ObbConversionError(f"Expected 4 corners, got shape {corners_norm.shape}")
        coords = " ".join(f"{v:.6f}" for v in corners_norm.flatten())
        return f"{class_id} {coords}"
