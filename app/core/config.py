"""Application configuration settings."""

import os
from pathlib import Path
from typing import List

class Settings:
    PROJECT_NAME: str = "InmobiliariaVR - Vision Backend"
    VERSION: str = "0.2.0"
    API_V1_STR: str = "/api/v1"

    # Server configuration
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    DEBUG: bool = os.getenv("DEBUG", "true").lower() in ("true", "1")

    # CORS
    CORS_ORIGINS: List[str] = ["*"]

    # Upload limits
    MAX_IMAGE_SIZE_BYTES: int = 15 * 1024 * 1024  # 15 MB
    ALLOWED_EXTENSIONS: set = {"jpg", "jpeg", "png", "webp"}

    # Base paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    TEMP_DIR: Path = BASE_DIR / "temp"

    # Furniture & Architectural element detection (Scene Compiler / Fase 2A).
    # Unified YOLOv8s-seg model trained on CubiCasa5K (real SVG) + FloorPlanCAD,
    # 29 classes (see YOLO/unified_train data.yaml / YOLO/docs/05_*.md).
    # Furniture from FloorPlanCAD (bed/sofa/table/chair/refrigerator) validates
    # at mAP50 ~0.75-0.9; toilet/sink/window are weak (~0.05-0.23).
    YOLO_MODEL_PATH: str = os.getenv("YOLO_MODEL_PATH", "YOLO/best_unified_seg.pt")
    # Tipo de modelo: "detect" (bboxes), "obb" (oriented bboxes), "seg" (segmentation polygons)
    YOLO_MODEL_TYPE: str = os.getenv("YOLO_MODEL_TYPE", "seg")
    # Inference resolution. Structural classes/fixtures do best at the training
    # size (640); movable furniture (read from the inverted pass) does best at a
    # larger size on dense plans -- measured on the sample CAD plans.
    YOLO_IMAGE_SIZE: int = int(os.getenv("YOLO_IMAGE_SIZE", "640"))
    YOLO_FURNITURE_IMAGE_SIZE: int = int(os.getenv("YOLO_FURNITURE_IMAGE_SIZE", "1280"))
    YOLO_CONFIDENCE_THRESHOLD: float = float(os.getenv("YOLO_CONFIDENCE_THRESHOLD", "0.25"))

    # Fallback room size (meters) used when OCR can't read a written dimension
    # from the sketch. Matches Sala_MVP's default room size.
    DEFAULT_ROOM_WIDTH_M: float = 4.0
    DEFAULT_ROOM_DEPTH_M: float = 4.0
    DEFAULT_ROOM_HEIGHT_M: float = 2.5

settings = Settings()
