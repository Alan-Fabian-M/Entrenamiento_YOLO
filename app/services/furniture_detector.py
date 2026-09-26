"""Furniture detection service ("the eyes" of the pipeline).

Ported and generalized from the original `SpatialSceneCompiler/scene_compiler.py`
prototype (validated end-to-end on 2026-09-10 with a real photo). Three model types
are supported:

1. **Detection** (YOLOv8-det): axis-aligned bounding boxes, angle always 0
2. **Oriented Bounding Box** (YOLOv8-OBB): rotated boxes with real orientation from xywhr
3. **Segmentation** (YOLOv8-seg): polygon masks with centroid position and rotation
   from oriented minimum rectangle. Used with CubiCasa5k dataset for accurate furniture shapes.

The model type is controlled by `settings.YOLO_MODEL_TYPE` ("detect", "obb", "seg").
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np
from shapely.geometry import Polygon
from ultralytics import YOLO

from app.core.config import settings

logger = logging.getLogger(__name__)

# Mapping of dataset class names -> unified type understood by Unity.
# Covers the current unified model (29 classes, CubiCasa5K + FloorPlanCAD, see
# YOLO/docs/05_*.md) plus the legacy COCO / FloorPlanCAD names so older weights
# keep working.
CLASS_TO_UNITY_TYPE = {
    # COCO mappings
    "couch": "sofa",
    "dining table": "mesa",
    "tv": "tv",
    "oven": "horno",
    "microwave": "microondas",
    # Legacy FloorPlanCAD names
    "single_door": "puerta",
    "double_door": "puerta_doble",
    "sliding_door": "puerta_corrediza",
    "stair": "escalera",
    "bath_tub": "tina",
    # Shared by every model (legacy + unified)
    "bed": "cama",
    "sofa": "sofa",
    "table": "mesa",
    "chair": "silla",
    "desk": "escritorio",
    "door": "puerta",
    "window": "ventana",
    "column": "columna",
    "stairs": "escaleras",
    "kitchen_counter": "mostrador_cocina",
    "toilet": "inodoro",
    "sink": "lavabo",
    "bathtub": "tina",
    "wardrobe": "armario",
    "refrigerator": "refrigerador",
    "gas_stove": "estufa",
    # Unified model additions
    "shower": "ducha",
    "fireplace": "chimenea",
    # NB: intentionally NOT mapped:
    #  - "wall": the room boundary is emitted by RoomExtractor as structural
    #    muros; per-segment YOLO walls would duplicate/conflict with those.
    #  - "room", "bathroom", "kitchen", "living_room", "bedroom", "dining_room",
    #    "outdoor": areas, not furniture (candidates to feed room_type later).
    #  - "cabinet", "electrical_appliance", "bench", "misc_furniture": weak in
    #    validation (mAP50 < 0.1) and no known Unity prefab; add once verified.
}
COCO_CLASS_TO_UNITY_TYPE = CLASS_TO_UNITY_TYPE

# Classes only learned from FloorPlanCAD's dark-background renders.
_INVERTED_ONLY_CLASSES = {"bed", "sofa", "table", "chair", "refrigerator", "gas_stove"}


@dataclass
class DetectedFurniture:
    unity_type: str
    confidence: float
    center_px: Tuple[float, float]
    size_px: Tuple[float, float]
    angle_deg: float
    color_hex: str


def _box_iou(a: DetectedFurniture, b: DetectedFurniture) -> float:
    """IoU of the axis-aligned boxes implied by center/size (rotation ignored)."""
    ax0, ay0 = a.center_px[0] - a.size_px[0] / 2, a.center_px[1] - a.size_px[1] / 2
    ax1, ay1 = a.center_px[0] + a.size_px[0] / 2, a.center_px[1] + a.size_px[1] / 2
    bx0, by0 = b.center_px[0] - b.size_px[0] / 2, b.center_px[1] - b.size_px[1] / 2
    bx1, by1 = b.center_px[0] + b.size_px[0] / 2, b.center_px[1] + b.size_px[1] / 2
    iw, ih = min(ax1, bx1) - max(ax0, bx0), min(ay1, by1) - max(ay0, by0)
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    union = a.size_px[0] * a.size_px[1] + b.size_px[0] * b.size_px[1] - inter
    return inter / union if union > 0 else 0.0


def suppress_overlaps(
    detections: List[DetectedFurniture], iou_threshold: float = 0.5
) -> List[DetectedFurniture]:
    """Class-agnostic NMS: YOLO's own NMS is per-class, so one object can come
    back as e.g. both `sofa` and `cama`. Keep the most confident of any
    overlapping pair."""
    kept: List[DetectedFurniture] = []
    for det in sorted(detections, key=lambda d: d.confidence, reverse=True):
        if all(_box_iou(det, k) < iou_threshold for k in kept):
            kept.append(det)
    return kept


class FurnitureDetector:
    """Wraps an Ultralytics YOLO model (regular or OBB) behind a stable interface."""

    _model: YOLO | None = None
    _model_path: str | None = None

    @classmethod
    def _get_model(cls) -> YOLO:
        if cls._model is None or cls._model_path != settings.YOLO_MODEL_PATH:
            logger.info("Loading YOLO model: %s", settings.YOLO_MODEL_PATH)
            cls._model = YOLO(settings.YOLO_MODEL_PATH)
            cls._model_path = settings.YOLO_MODEL_PATH
        return cls._model

    @staticmethod
    def _average_color_hex(image_bgr: np.ndarray, bbox_xyxy: Tuple[float, float, float, float]) -> str:
        """Approximate dominant color (mean pixel value) of the detected crop."""
        x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, image_bgr.shape[1]), min(y2, image_bgr.shape[0])
        if x2 <= x1 or y2 <= y1:
            return "#CCCCCC"

        crop = image_bgr[y1:y2, x1:x2]
        b, g, r = [int(round(c)) for c in cv2.mean(crop)[:3]]
        return "#{:02X}{:02X}{:02X}".format(r, g, b)

    @staticmethod
    def _polygon_to_furniture(
        polygon_xy: np.ndarray,
        class_name: str,
        confidence: float,
        image_bgr: np.ndarray,
    ) -> DetectedFurniture | None:
        """Convert a segmentation polygon (already in pixel coords, as returned
        by Ultralytics' `mask.xy`) to DetectedFurniture.

        Extracts centroid for position, minimum rotated rectangle for rotation angle,
        and bounding box for size. Returns None if polygon is invalid.
        """
        if len(polygon_xy) < 3:
            return None

        # polygon_xy from mask.xy is already in pixel coordinates -- do NOT
        # multiply by (w, h) again (that's only needed for mask.xyn, the
        # normalized 0-1 variant, which we don't use here).
        polygon_px = np.asarray(polygon_xy, dtype=np.float64)

        try:
            poly = Polygon(polygon_px)
            if not poly.is_valid or poly.area < 10:  # Skip tiny detections
                return None

            # Centroid for position
            centroid = poly.centroid
            cx, cy = centroid.x, centroid.y

            # Minimum rotated rectangle for rotation
            coords = np.array(poly.exterior.coords[:-1])  # Exclude closing point
            if len(coords) < 3:
                return None

            rect = cv2.minAreaRect(coords.astype(np.float32))
            angle_deg = rect[2]  # Rotation angle in degrees
            if angle_deg < -45:
                angle_deg += 90

            # Bounding box for size
            (bbox_x, bbox_y), (bbox_w, bbox_h) = rect[:2]

            # Bounded color sample from centroid region
            bbox_int = poly.bounds  # (minx, miny, maxx, maxy)
            x1, y1, x2, y2 = [int(v) for v in bbox_int]
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = min(x2, image_bgr.shape[1]), min(y2, image_bgr.shape[0])

            if x2 <= x1 or y2 <= y1:
                color_hex = "#CCCCCC"
            else:
                crop = image_bgr[y1:y2, x1:x2]
                b, g, r = [int(round(c)) for c in cv2.mean(crop)[:3]]
                color_hex = "#{:02X}{:02X}{:02X}".format(r, g, b)

            unity_type = COCO_CLASS_TO_UNITY_TYPE.get(class_name)
            if unity_type is None:
                return None

            return DetectedFurniture(
                unity_type=unity_type,
                confidence=confidence,
                center_px=(cx, cy),
                size_px=(bbox_w, bbox_h),
                angle_deg=angle_deg,
                color_hex=color_hex,
            )
        except Exception as e:
            logger.warning(f"Failed to process polygon for {class_name}: {e}")
            return None

    @classmethod
    def detect(cls, image_bgr: np.ndarray) -> List[DetectedFurniture]:
        """Detect furniture instances. Returns unified-type detections with pixel
        coordinates; the caller is responsible for converting px -> meters.

        Supports three model types via settings.YOLO_MODEL_TYPE:
        - "detect": YOLOv8-det (axis-aligned bboxes, angle=0)
        - "obb": YOLOv8-OBB (rotated boxes, real angles from xywhr)
        - "seg": YOLOv8-seg (polygons, centroid position, rotation from min rect)
        """
        model = cls._get_model()
        model_type = settings.YOLO_MODEL_TYPE.lower()

        # IMPORTANT: Ultralytics defaults to an internal conf=0.25 cutoff when
        # `conf` isn't passed explicitly, applied *before* results are returned.
        # Our own settings.YOLO_CONFIDENCE_THRESHOLD filter below only sees
        # whatever survives that internal cutoff, so it must be passed here
        # too -- otherwise lowering settings.YOLO_CONFIDENCE_THRESHOLD has no
        # effect and low-confidence detections never reach our filter at all.
        # Domain gap: FloorPlanCAD (the only source of bed/sofa/table/chair/
        # refrigerator/gas_stove) is rendered light-lines-on-BLACK, while user
        # plans are dark-lines-on-white (like CubiCasa5K). Those classes are
        # therefore read from an inverted copy of light-background plans; all
        # other classes come from the original image. Geometry is unchanged.
        passes: List[Tuple[np.ndarray, callable, int]] = [
            (image_bgr, lambda name: True, settings.YOLO_IMAGE_SIZE)
        ]
        if _INVERTED_ONLY_CLASSES <= set(model.names.values()) and cv2.cvtColor(
            image_bgr, cv2.COLOR_BGR2GRAY
        ).mean() > 127:
            passes = [
                (image_bgr, lambda name: name not in _INVERTED_ONLY_CLASSES, settings.YOLO_IMAGE_SIZE),
                (
                    cv2.bitwise_not(image_bgr),
                    lambda name: name in _INVERTED_ONLY_CLASSES,
                    settings.YOLO_FURNITURE_IMAGE_SIZE,
                ),
            ]

        pass_results = []
        for pass_image, allowed, imgsz in passes:
            for result in model(
                pass_image,
                conf=settings.YOLO_CONFIDENCE_THRESHOLD,
                imgsz=imgsz,
                verbose=False,
            ):
                pass_results.append((result, allowed))
        detections: List[DetectedFurniture] = []

        for result, allowed in pass_results:
            names = result.names

            # ===== Segmentation model (YOLOv8-seg): polygon masks =====
            if model_type == "seg" and result.masks is not None:
                for i, mask in enumerate(result.masks):
                    if result.boxes is None or i >= len(result.boxes):
                        continue

                    box = result.boxes[i]
                    class_id = int(box.cls[0])
                    coco_name = names[class_id]
                    if not allowed(coco_name):
                        continue
                    confidence = float(box.conf[0])

                    if confidence < settings.YOLO_CONFIDENCE_THRESHOLD:
                        continue

                    # mask.xy returns a LIST with one array of (x, y) points
                    # per instance, already in pixel coordinates (NOT
                    # normalized -- that's mask.xyn). Must unwrap the list
                    # to get the actual point array before checking length,
                    # otherwise len() always measures the outer list (== 1)
                    # and every detection gets skipped regardless of shape.
                    polygon_xy_list = mask.xy
                    if not polygon_xy_list or len(polygon_xy_list[0]) < 3:
                        continue
                    polygon_px = polygon_xy_list[0]

                    furniture = cls._polygon_to_furniture(polygon_px, coco_name, confidence, image_bgr)
                    if furniture is not None:
                        detections.append(furniture)

            # ===== OBB model (YOLOv8-OBB): rotated boxes =====
            elif model_type == "obb" and result.obb is not None:
                for box in result.obb:
                    coco_name = names[int(box.cls[0])]
                    if not allowed(coco_name):
                        continue
                    unity_type = COCO_CLASS_TO_UNITY_TYPE.get(coco_name)
                    confidence = float(box.conf[0])
                    if unity_type is None or confidence < settings.YOLO_CONFIDENCE_THRESHOLD:
                        continue

                    cx, cy, w, h, angle_rad = [float(v) for v in box.xywhr[0]]
                    angle_deg = np.degrees(angle_rad)
                    x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2

                    detections.append(
                        DetectedFurniture(
                            unity_type=unity_type,
                            confidence=confidence,
                            center_px=(cx, cy),
                            size_px=(w, h),
                            angle_deg=angle_deg,
                            color_hex=cls._average_color_hex(image_bgr, (x1, y1, x2, y2)),
                        )
                    )

            # ===== Detection model (YOLOv8-det): axis-aligned boxes (default) =====
            else:
                if result.boxes is None:
                    continue
                for box in result.boxes:
                    coco_name = names[int(box.cls[0])]
                    if not allowed(coco_name):
                        continue
                    unity_type = COCO_CLASS_TO_UNITY_TYPE.get(coco_name)
                    confidence = float(box.conf[0])
                    if unity_type is None or confidence < settings.YOLO_CONFIDENCE_THRESHOLD:
                        continue

                    x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
                    detections.append(
                        DetectedFurniture(
                            unity_type=unity_type,
                            confidence=confidence,
                            center_px=((x1 + x2) / 2.0, (y1 + y2) / 2.0),
                            size_px=(x2 - x1, y2 - y1),
                            # Detection model: no real orientation available.
                            # User adjusts rotation by hand in VR.
                            angle_deg=0.0,
                            color_hex=cls._average_color_hex(image_bgr, (x1, y1, x2, y2)),
                        )
                    )

        return suppress_overlaps(detections)
