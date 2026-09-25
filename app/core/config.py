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
    # Modelo entrenado con CubiCasa5k (5000+ planos CAD reales con anotaciones de segmentación).
    # Clases: bed, sofa, table, chair, desk, door, window, wall, column, stairs,
    #         kitchen_counter, toilet, sink, bathtub, wardrobe (15 clases objetivo)
    YOLO_MODEL_PATH: str = os.getenv("YOLO_MODEL_PATH", "YOLO/best_seg.pt")
    # Tipo de modelo: "detect" (bboxes), "obb" (oriented bboxes), "seg" (segmentation polygons)
    YOLO_MODEL_TYPE: str = os.getenv("YOLO_MODEL_TYPE", "seg")
    # NOTE: current best_seg.pt has weak confidence calibration (low training
    # mAP) -- real detections top out around 0.10-0.15. Once retrained on the
    # full real CubiCasa5k dataset this should go back up toward 0.35+.
    YOLO_CONFIDENCE_THRESHOLD: float = float(os.getenv("YOLO_CONFIDENCE_THRESHOLD", "0.05"))

    # Fallback room size (meters) used when OCR can't read a written dimension
    # from the sketch. Matches Sala_MVP's default room size.
    DEFAULT_ROOM_WIDTH_M: float = 4.0
    DEFAULT_ROOM_DEPTH_M: float = 4.0
    DEFAULT_ROOM_HEIGHT_M: float = 2.5

settings = Settings()
