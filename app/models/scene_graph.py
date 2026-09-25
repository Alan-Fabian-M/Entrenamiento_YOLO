"""Pydantic models for the Scene Graph JSON contract consumed by Unity.

This is the SAME schema that `SceneGenerator.cs` / `SceneValidator.cs` on the
Unity side already parse (and that the bundled sample scenes under
`Assets/Resources/ScenePresets/*.json` already use), documented in the
InmobiliariaVR project as `json-schema-contrato-escenas.md`. Keeping this
model in sync with that document -- and with Unity's `PrefabMapper.cs` type
list -- is what lets `/api/v1/compilar-sala` be consumed directly by
`CroquisSceneCompilerController.cs` without any translation layer on the
Unity side.

Do NOT rename fields here without updating both the Unity C# side and the
markdown contract doc in the same change. New fields must always ship with a
default so a response built without setting them still round-trips (e.g.
`rooms`, `room_id`, `room_count` below) -- Unity's JSON parsing is assumed to
ignore unrecognized fields, but this hasn't been verified from this repo, so
staying additive-only is what keeps that assumption safe either way.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional

from pydantic import BaseModel, Field


class Vector3(BaseModel):
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


class ElementMaterial(BaseModel):
    color: str = Field(description="Hex color, e.g. '#8899AA'")
    type: Optional[str] = Field(default=None, description="Cosmetic hint only (e.g. 'fabric', 'wood')")


class ElementMetadataFlags(BaseModel):
    """Optional per-element interaction overrides.

    Left unset on purpose for most elements: Unity's `PrefabMapper.GetMetadata()`
    already has sensible per-type defaults (walls are fixed, furniture is
    selectable/movable/colorable/deletable, etc.), so duplicating that logic
    here would just be two places that can drift out of sync.
    """

    selectable: Optional[bool] = None
    deletable: Optional[bool] = None
    colorable: Optional[bool] = None
    movable: Optional[bool] = None


class ElementTransform(BaseModel):
    position: Vector3
    rotation: Vector3 = Field(default_factory=Vector3)
    scale: Vector3 = Field(default_factory=lambda: Vector3(x=1.0, y=1.0, z=1.0))


class ElementProperties(BaseModel):
    color_hex: Optional[str] = None
    material_type: Optional[str] = None
    room_id: Optional[str] = Field(default=None, description="Which detected room this element belongs to, e.g. 'room_0'. Unset for single-room responses.")


class SceneElement(BaseModel):
    id: str
    type: str = Field(description="Furniture/structural type. See PrefabMapper.cs for the supported list.")
    class_label: Optional[str] = None
    prefab_id: Optional[str] = None
    confidence: float = Field(default=0.85, ge=0.0, le=1.0)
    position: Vector3
    rotation: Vector3 = Field(default_factory=Vector3)
    scale: Vector3 = Field(default_factory=lambda: Vector3(x=1.0, y=1.0, z=1.0))
    material: Optional[ElementMaterial] = None
    metadata: Optional[ElementMetadataFlags] = None
    transform: Optional[ElementTransform] = None
    properties: Optional[ElementProperties] = None


class RoomDimensions(BaseModel):
    x: float
    y: float
    z: float


class RoomInfo(BaseModel):
    room_type: str = "monoambiente"
    dimensions: RoomDimensions
    flooring_material: str = "wood"
    wall_color: str = "white"


class RoomShapeInfo(BaseModel):
    """One detected room among possibly several. `room_info` (singular, above)
    stays a backward-compatible summary of the whole floor plan's union
    bounding box; this is the new, additive per-room detail.
    """

    room_id: str
    room_type: str = "ambiente"
    dimensions: RoomDimensions
    center: Vector3


class SceneMetadataDimensions(BaseModel):
    width: float
    depth: float
    height: float


class SceneMetadata(BaseModel):
    name: str = "Sala generada desde croquis"
    description: str = "Generado automaticamente por el Spatial Scene Compiler a partir de una foto"
    dimensions: SceneMetadataDimensions
    furniture_count: int
    created_date: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    processing_time_ms: int = 0
    scale_confidence: float = Field(default=0.0, description="0 means no dimension text was OCR'd; a default room size was assumed.")
    scale_source: str = ""
    room_count: int = Field(default=1, description="Number of rooms detected; 1 for the single-room fallback path.")


class SceneGraphResponse(BaseModel):
    """Top-level response for POST /api/v1/compilar-sala.

    This exact shape is what `SceneGenerator.GenerateSceneAsync()` expects.
    """

    metadata: SceneMetadata
    room_info: RoomInfo
    rooms: List[RoomShapeInfo] = Field(default_factory=list, description="Per-room detail; empty for the single-room fallback path.")
    scene_elements: List[SceneElement]
