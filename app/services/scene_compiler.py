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
from shapely.geometry import Point, Polygon

from app.core.config import settings
from app.models.scene_graph import (
    ElementMaterial,
    ElementProperties,
    ElementTransform,
    RoomDimensions,
    RoomInfo,
    RoomShapeInfo,
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

# Priority-ordered keyword rules for inferring a room's likely use from the
# furniture assigned to it. Checked in order; first match wins.
_ROOM_TYPE_RULES: list[tuple[str, set[str]]] = [
    ("baño", {"inodoro", "lavabo", "tina"}),
    ("cocina", {"estufa", "horno", "refrigerador", "mostrador_cocina", "microondas"}),
    ("dormitorio", {"cama"}),
    ("sala", {"sofa", "tv"}),
    ("comedor", {"mesa", "silla"}),
]


def _infer_room_type(furniture_types: set[str]) -> str:
    for room_type, keywords in _ROOM_TYPE_RULES:
        if furniture_types & keywords:
            return room_type
    return "ambiente"


class SceneCompiler:
    @classmethod
    def compile(cls, image_bytes: bytes) -> SceneGraphResponse:
        t0 = time.perf_counter()

        bgr = ImagePreprocessor.decode_image(image_bytes)
        image_h_px, image_w_px = bgr.shape[:2]

        # YOLO-seg (CubiCasa5k) was trained on clean CAD renders, not on
        # shadow-removed/contrast-enhanced photos. The shadow-removal +
        # contrast-enhancement pipeline was tuned for real photographs and
        # was destroying the CAD line work, driving detection confidences
        # to near-zero. Feed YOLO the raw image directly (matches the
        # image the model was validated against in debug_model.py).
        # If photo-style sketches need this preprocessing again later,
        # gate it behind an explicit "input is a photo" flag instead of
        # always applying it.
        furniture = FurnitureDetector.detect(bgr)
        rooms = RoomExtractor.detect_rooms(bgr)

        if rooms:
            # Shared origin/size = the union bounding box of every detected
            # room. Everything below (scale, furniture positions, per-room
            # meters) is computed relative to THIS shared frame, not each
            # room's own frame -- mixing the two is the easiest way to
            # silently misplace room 2 relative to room 1.
            xs0 = [r.origin_px[0] for r in rooms]
            ys0 = [r.origin_px[1] for r in rooms]
            xs1 = [r.origin_px[0] + r.width_px for r in rooms]
            ys1 = [r.origin_px[1] + r.height_px for r in rooms]
            origin_px = (min(xs0), min(ys0))
            room_width_px = max(xs1) - origin_px[0]
            room_height_px = max(ys1) - origin_px[1]
        else:
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
        else:
            # No reliable OCR'd scale: fall back to a sane default room width
            # (same default as Sala_MVP) instead of inventing odd numbers, and
            # report confidence 0 so the caller can ask the user to confirm.
            # Depth (and every per-room size) still derives from real detected
            # pixel geometry via px_per_m below -- never hardcoded independently.
            px_per_m = room_width_px / settings.DEFAULT_ROOM_WIDTH_M if room_width_px else 1.0

        width_m = round(room_width_px / px_per_m, 2)
        depth_m = round(room_height_px / px_per_m, 2)

        # room_id -> (RoomShape, shapely Polygon, (center_x_m, center_z_m), width_m, depth_m)
        room_index: dict = {}

        if rooms:
            scene_elements = []
            for room in rooms:
                r_width_m = round(room.width_px / px_per_m, 2)
                r_depth_m = round(room.height_px / px_per_m, 2)

                center_px_x = room.origin_px[0] + room.width_px / 2.0 - origin_px[0]
                center_px_y = room.origin_px[1] + room.height_px / 2.0 - origin_px[1]
                center_x_m = round(center_px_x / px_per_m, 2)
                center_z_m = round((room_height_px - center_px_y) / px_per_m, 2)

                scene_elements.extend(
                    RoomExtractor.build_room_walls(
                        room,
                        width_m=r_width_m,
                        depth_m=r_depth_m,
                        height_m=settings.DEFAULT_ROOM_HEIGHT_M,
                        center_m=(center_x_m, center_z_m),
                    )
                )

                coords = room.contour_px
                if len(coords) < 3:
                    x, y = room.origin_px
                    w, h = room.width_px, room.height_px
                    coords = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
                polygon = Polygon(coords)
                if not polygon.is_valid:
                    polygon = polygon.buffer(0)

                room_index[room.room_id] = (room, polygon, (center_x_m, center_z_m), r_width_m, r_depth_m)
        else:
            scene_elements = RoomExtractor.build_wall_elements(
                width_m=width_m, depth_m=depth_m, height_m=settings.DEFAULT_ROOM_HEIGHT_M
            )

        room_furniture_types: dict = {room_id: set() for room_id in room_index}

        for i, item in enumerate(furniture):
            px, py = item.center_px

            room_id = None
            if room_index:
                point = Point(px, py)
                for candidate_id, (_room, polygon, *_rest) in room_index.items():
                    if polygon.is_valid and polygon.contains(point):
                        room_id = candidate_id
                        break
                if room_id is None:
                    room_id = min(
                        room_index,
                        key=lambda rid: point.distance(room_index[rid][1].centroid),
                    )
                room_furniture_types[room_id].add(item.unity_type)

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
                        room_id=room_id,
                    ),
                ).model_dump()
            )

        rooms_out = [
            RoomShapeInfo(
                room_id=room_id,
                room_type=_infer_room_type(room_furniture_types[room_id]),
                dimensions=RoomDimensions(x=r_width_m, y=settings.DEFAULT_ROOM_HEIGHT_M, z=r_depth_m),
                center=Vector3(x=center_m[0], y=0.0, z=center_m[1]),
            )
            for room_id, (_room, _polygon, center_m, r_width_m, r_depth_m) in room_index.items()
        ]

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
                room_count=max(len(rooms_out), 1),
            ),
            room_info=RoomInfo(
                room_type="detectado_desde_croquis" if len(rooms_out) <= 1 else "multiple",
                dimensions=RoomDimensions(x=width_m, y=settings.DEFAULT_ROOM_HEIGHT_M, z=depth_m),
            ),
            rooms=rooms_out,
            scene_elements=scene_elements,
        )
