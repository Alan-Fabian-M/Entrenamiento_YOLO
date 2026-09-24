"""Scene Graph orchestrator: photo -> JSON that `SceneGenerator.cs` consumes.

Ties together the three perception channels (furniture via YOLO, room bounds
via OpenCV, scale via OCR) plus the preprocessing service, and assembles the
result in the schema documented in `json-schema-contrato-escenas.md`
(the InmobiliariaVR Claude project) -- the SAME schema already used by
`Assets/Resources/ScenePresets/preset_*.json` and consumed by
`SceneGenerator.GenerateSceneAsync()` on the Unity side.
"""

from __future__ import annotations

import time

import numpy as np

from app.core.config import settings
from app.models.scene_graph import (
    ElementMaterial,
    ElementProperties,
    ElementTransform,
    RoomDimensions,
    RoomInfo,
    SceneElement,
    SceneGraphResponse,
    SceneMetadata,
    SceneMetadataDimensions,
    Vector3,
)
from app.services.furniture_detector import FurnitureDetector
from app.services.preprocessor import ImagePreprocessor
from app.services.room_extractor import RoomExtractor
from app.services.scale_detector import ScaleDetector


class SceneCompiler:
    @classmethod
    def compile(cls, image_bytes: bytes) -> SceneGraphResponse:
        t0 = time.perf_counter()

        bgr = ImagePreprocessor.decode_image(image_bytes)
        image_h_px, image_w_px = bgr.shape[:2]

        # YOLO gets the shadow-free + contrast-enhanced variant (best for detection);
        # OpenCV contour + OCR get the plain image, same as the validated prototype.
        gray = ImagePreprocessor.to_grayscale(bgr)
        shadow_free_gray = ImagePreprocessor.enhance_contrast(ImagePreprocessor.remove_shadows(gray))
        shadow_free_bgr = np.stack([shadow_free_gray] * 3, axis=-1)

        furniture = FurnitureDetector.detect(shadow_free_bgr)
        room_contour = RoomExtractor.detect_bounds(bgr)

        if room_contour:
            room_width_px, room_height_px = room_contour.width_px, room_contour.height_px
            origin_px = room_contour.origin_px
        else:
            room_width_px, room_height_px = image_w_px, image_h_px
            origin_px = (0, 0)

        scale = ScaleDetector.detect(bgr, room_width_px)

        if scale.pixels_per_meter:
            px_per_m = scale.pixels_per_meter
            width_m = round(room_width_px / px_per_m, 2)
            depth_m = round(room_height_px / px_per_m, 2)
        else:
            # No reliable OCR'd scale: fall back to a sane default room size
            # (same default as Sala_MVP) instead of inventing odd numbers, and
            # report confidence 0 so the caller can ask the user to confirm.
            width_m, depth_m = settings.DEFAULT_ROOM_WIDTH_M, settings.DEFAULT_ROOM_DEPTH_M
            px_per_m = room_width_px / width_m if room_width_px else 1.0

        scene_elements = RoomExtractor.build_wall_elements(
            width_m=width_m, depth_m=depth_m, height_m=settings.DEFAULT_ROOM_HEIGHT_M
        )

        for i, item in enumerate(furniture):
            px, py = item.center_px
            x_m = round((px - origin_px[0]) / px_per_m, 2)
            z_m = round((room_height_px - (py - origin_px[1])) / px_per_m, 2)  # image Y -> Unity Z (inverted)

            pos_v3 = Vector3(x=x_m, y=0.0, z=z_m)
            rot_v3 = Vector3(x=0.0, y=item.angle_deg, z=0.0)
            scale_v3 = Vector3(x=1.0, y=1.0, z=1.0)

            scene_elements.append(
                SceneElement(
                    id=f"obj_{i:03d}",
                    type=item.unity_type,
                    class_label=item.unity_type,
                    prefab_id=item.unity_type,
                    confidence=round(item.confidence, 2),
                    position=pos_v3,
                    rotation=rot_v3,
                    scale=scale_v3,
                    material=ElementMaterial(color=item.color_hex),
                    transform=ElementTransform(
                        position=pos_v3,
                        rotation=rot_v3,
                        scale=scale_v3,
                    ),
                    properties=ElementProperties(
                        color_hex=item.color_hex,
                    ),
                ).model_dump()
            )

        processing_time_ms = round((time.perf_counter() - t0) * 1000)

        return SceneGraphResponse(
            metadata=SceneMetadata(
                name="Sala generada desde croquis",
                description="Compilado automaticamente por el Spatial Scene Compiler a partir de una foto",
                dimensions=SceneMetadataDimensions(width=width_m, depth=depth_m, height=settings.DEFAULT_ROOM_HEIGHT_M),
                furniture_count=len(furniture),
                processing_time_ms=processing_time_ms,
                scale_confidence=scale.confidence,
                scale_source=scale.source,
            ),
            room_info=RoomInfo(
                room_type="detectado_desde_croquis",
                dimensions=RoomDimensions(x=width_m, y=settings.DEFAULT_ROOM_HEIGHT_M, z=depth_m),
            ),
            scene_elements=scene_elements,
        )
