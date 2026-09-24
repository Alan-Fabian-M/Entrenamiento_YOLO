"""Maps CubiCasa5K raw SVG `class` annotations to a unified YOLOv8-OBB taxonomy."""

import logging
import re
from typing import Optional, Set

logger = logging.getLogger(__name__)

# Order matters: this is the fixed class_id ordering for YOLO labels.
# Do NOT reorder once training has started against a generated dataset.
YOLO_CLASSES = [
    "door",               # 0
    "window",             # 1
    "sofa",               # 2
    "bed",                # 3
    "table",              # 4
    "chair",              # 5
    "toilet",             # 6
    "sink",               # 7
    "bathtub",            # 8
    "shower",             # 9
    "cabinet",            # 10
    "wardrobe",           # 11
    "stairs",             # 12
    "fireplace",          # 13
    "kitchen_appliance",  # 14
    "electrical",         # 15
    "counter_top",        # 16
    "misc_furniture",     # 17
]

# Raw CubiCasa5K SVG class strings (normalized: lowercased, whitespace-collapsed)
# mapped to one of the unified YOLO_CLASSES above. This is a best-effort initial
# taxonomy based on the publicly documented CubiCasa5K icon set; refine it with
# `python -m training.cli audit-classes` once the real dataset is available.
RAW_TO_UNIFIED = {
    # Doors
    "door": "door",
    "sliding door": "door",
    "folding door": "door",
    "revolving door": "door",
    "door 1-panel": "door",
    "door 2-panel": "door",
    # Windows
    "window": "window",
    "sliding window": "window",
    "corner window": "window",
    # Seating / living room
    "sofa": "sofa",
    "chaise longue": "sofa",
    "armchair": "sofa",
    # Bedroom
    "bed": "bed",
    # Tables
    "table": "table",
    "dining table": "table",
    "coffee table": "table",
    "chair": "chair",
    # Bathroom fixtures
    "toilet": "toilet",
    "sink": "sink",
    "double sink": "sink",
    "bathtub": "bathtub",
    "shower": "shower",
    # Storage
    "kitchen cabinet": "cabinet",
    "cabinet": "cabinet",
    "wardrobe": "wardrobe",
    "closet": "wardrobe",
    # Circulation
    "stairs": "stairs",
    "railing": "stairs",
    # Fireplace / chimney
    "chimney": "fireplace",
    "fireplace": "fireplace",
    "sauna bench": "fireplace",
    # Kitchen appliances
    "stove": "kitchen_appliance",
    "gas stove": "kitchen_appliance",
    "oven": "kitchen_appliance",
    "dishwasher": "kitchen_appliance",
    "fridge": "kitchen_appliance",
    "refrigerator": "kitchen_appliance",
    "washing machine": "kitchen_appliance",
    # Electrical
    "electrical appliance": "electrical",
    # Counters
    "counter top": "counter_top",
    "kitchen counter": "counter_top",
}

# Raw classes explicitly excluded from YOLO-OBB training: they belong to the
# Phase 3 geometric engine (wall/room reconstruction), not to object detection.
EXCLUDED_RAW_CLASSES: Set[str] = {
    "wall",
    "room",
    "background",
    "column",
    "floor",
    "space",
    "outer wall",
    "inner wall",
}

_WHITESPACE_RE = re.compile(r"\s+")

# Tracks raw classes reported as unrecognized this run, so warnings are logged once.
_reported_unrecognized: Set[str] = set()


class ClassMappingError(Exception):
    """Raised for invalid class-mapping lookups."""
    pass


class ClassMapper:
    """Normalizes raw CubiCasa5K SVG class strings and maps them to YOLO classes."""

    @staticmethod
    def normalize_raw_class(raw: str) -> str:
        """Lowercase, strip, and collapse internal whitespace."""
        return _WHITESPACE_RE.sub(" ", raw.strip().lower())

    @classmethod
    def map_to_unified(cls, raw_class: str) -> Optional[str]:
        """Return the unified YOLO class name, or None if `raw_class` should be
        excluded (wall/room/etc.) or is unrecognized. Unrecognized classes are
        logged once per run for later taxonomy iteration via `audit-classes` -
        they are never silently assigned to `misc_furniture`.
        """
        normalized = cls.normalize_raw_class(raw_class)

        if normalized in EXCLUDED_RAW_CLASSES:
            return None

        if normalized in RAW_TO_UNIFIED:
            return RAW_TO_UNIFIED[normalized]

        if normalized not in _reported_unrecognized:
            _reported_unrecognized.add(normalized)
            logger.warning("Unrecognized CubiCasa5K raw class: %r", raw_class)

        return None

    @staticmethod
    def class_id(unified_class: str) -> int:
        """Return the fixed class_id for a unified YOLO class name."""
        try:
            return YOLO_CLASSES.index(unified_class)
        except ValueError as exc:
            raise ClassMappingError(f"Unknown unified class: {unified_class!r}") from exc

    @staticmethod
    def get_unrecognized_classes() -> Set[str]:
        """Return the set of raw classes seen but not recognized this run."""
        return set(_reported_unrecognized)

    @staticmethod
    def reset_unrecognized_classes() -> None:
        """Clear the unrecognized-classes tracker (mainly for test isolation)."""
        _reported_unrecognized.clear()
