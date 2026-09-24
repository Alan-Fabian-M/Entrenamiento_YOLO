"""Furniture detection service ("the eyes" of the pipeline).

Ported and generalized from the original `SpatialSceneCompiler/scene_compiler.py`
prototype (validated end-to-end on 2026-09-10 with a real photo). Two changes
from that prototype:

1. Output uses the *type* names from `PrefabMapper.cs` (the dynamic-instantiation
   pipeline), not the old `class_label`/`prefab_id` schema.
2. The model is swappable: today it defaults to the stock, COCO-pretrained
   `yolov8n.pt` (axis-aligned boxes only, angle always 0 -- same known
   limitation as the prototype, since there is no public YOLO-OBB checkpoint
   with furniture classes). Once `backend/training/cubicasa` produces a
   custom-trained YOLOv8-OBB checkpoint (see `training/cli.py build`, then
   train with `ultralytics`), point `settings.YOLO_MODEL_PATH` at those
   weights (e.g. `training/runs/obb/train/weights/best.pt`) and this service
   will automatically read real oriented angles from `box.xywhr` instead of
   defaulting to 0 -- no code changes needed elsewhere in the pipeline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from app.core.config import settings

logger = logging.getLogger(__name__)

# Mapping of dataset class names (COCO + FloorPlanCAD) -> unified type understood by Unity.
CLASS_TO_UNITY_TYPE = {
    # COCO mappings
    "couch": "sofa",
    "chair": "silla",
    "bed": "cama",
    "dining table": "mesa",
    "tv": "tv",
    "refrigerator": "refrigerador",
    "sink": "lavabo",
    "toilet": "inodoro",
    "oven": "horno",
    "microwave": "microondas",
    # FloorPlanCAD mappings
    "single_door": "puerta",
    "double_door": "puerta_doble",
    "sliding_door": "puerta_corrediza",
    "window": "ventana",
    "stair": "escalera",
    "sofa": "sofa",
    "table": "mesa",
    "bath_tub": "tina",
    "gas_stove": "estufa",
    "wardrobe": "armario",
    # NB: FloorPlanCAD's "wall" class is intentionally NOT mapped. Walls are
    # the room boundary and are emitted by RoomExtractor.build_wall_elements()
    # as the 4 structural muros; letting YOLO also emit per-segment "muro"
    # detections would duplicate/conflict with those in the Unity scene.
}
COCO_CLASS_TO_UNITY_TYPE = CLASS_TO_UNITY_TYPE


@dataclass
class DetectedFurniture:
    unity_type: str
    confidence: float
    center_px: Tuple[float, float]
    size_px: Tuple[float, float]
    angle_deg: float
    color_hex: str


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

    @classmethod
    def detect(cls, image_bgr: np.ndarray) -> List[DetectedFurniture]:
        """Detect furniture instances. Returns unified-type detections with pixel
        coordinates; the caller is responsible for converting px -> meters.
        """
        model = cls._get_model()
        is_obb_model = "-obb" in settings.YOLO_MODEL_PATH.lower() or "obb" in settings.YOLO_MODEL_PATH.lower()

        results = model(image_bgr, verbose=False)
        detections: List[DetectedFurniture] = []

        for result in results:
            names = result.names

            if is_obb_model and result.obb is not None:
                for box in result.obb:
                    coco_name = names[int(box.cls[0])]
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
            else:
                if result.boxes is None:
                    continue
                for box in result.boxes:
                    coco_name = names[int(box.cls[0])]
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
                            # No OBB model loaded -> no real orientation available.
                            # Same documented limitation as the original prototype:
                            # the user adjusts rotation by hand in VR.
                            angle_deg=0.0,
                            color_hex=cls._average_color_hex(image_bgr, (x1, y1, x2, y2)),
                        )
                    )

        return detections
