"""Maps CubiCasa5K raw SVG `class` annotations to a unified YOLOv8-seg taxonomy.

Rebuilt from REAL CubiCasa5K SVG data (not guessed): the class strings below
were collected by downloading the actual dataset (Zenodo record 2613548) and
grepping every distinct `class="..."` attribute across ~50 real `model.svg`
samples. CubiCasa5K only annotates FIXED fixtures (`FixedFurniture *`:
toilets, sinks, cabinets, closets, appliances...) -- it has NO icons for
movable furniture like beds, sofas, tables or chairs at all (this is a
building-permit floor plan dataset, not a furnishing dataset). Those classes
come from a separate dataset (FloorPlanCAD) merged in at training time by
`training/build_cubicasa_floorplancad_mix.py` -- do not add them here.
"""

import logging
import re
from typing import Optional, Set

logger = logging.getLogger(__name__)

# Order matters: this is the fixed class_id ordering for YOLO labels.
# Do NOT reorder once training has started against a generated dataset.
YOLO_CLASSES = [
    "wall",                 # 0
    "door",                 # 1
    "window",               # 2
    "column",               # 3
    "stairs",               # 4
    "room",                 # 5  generic/catch-all living space
    "bathroom",             # 6
    "kitchen",              # 7
    "living_room",          # 8
    "bedroom",              # 9
    "dining_room",          # 10
    "outdoor",              # 11  balcony/terrace/porch/garden
    "toilet",               # 12
    "sink",                 # 13
    "bathtub",              # 14
    "shower",                # 15
    "wardrobe",             # 16
    "cabinet",              # 17  kitchen/base/wall cabinets
    "kitchen_counter",      # 18
    "fireplace",            # 19
    "electrical_appliance", # 20  fridge/stove/dishwasher/washer/heater/...
    "bench",                # 21  incl. sauna benches
    "misc_furniture",       # 22  unrecognized-but-real FixedFurniture subtype
]

# Explicit mapping from normalized (lowercase, whitespace-collapsed) raw
# class strings -- as returned by CubiCasaSvgParser.walk_and_collect, which
# already resolves to the nearest SEMANTIC ancestor class -- to a unified
# YOLO_CLASSES name. Built from the real class vocabulary observed in the
# dataset; the prefix-based fallback in `map_to_unified` below covers any
# variant not explicitly listed here (the full 5000-sample dataset likely has
# a few more long-tail variants than the ~50-sample audit did).
RAW_TO_UNIFIED = {
    # Walls
    "wall": "wall",
    "wall external": "wall",
    # Doors (all variants: swing/slide/zfold/rollup/parallelslide, beside/opposite/none)
    "doors": "door",
    # Windows
    "window regular": "window",
    # Columns
    "column rectangle": "column",
    "column freeshape": "column",
    "column circle": "column",
    # Stairs / railings
    "stairs": "stairs",
    "railing": "stairs",
    # Room / space types (CubiCasa's "Space <Type>" groups)
    "space room": "room",
    "space undefined": "room",
    "space userdefined": "room",
    "space entry lobby": "room",
    "space entry": "room",
    "space draughtlobby": "room",
    "space hall": "room",
    "space closet walkin": "room",
    "space storage": "room",
    "space storage fuel": "room",
    "space technicalroom": "room",
    "space technicalroom boiler": "room",
    "space garage": "room",
    "space sauna": "room",
    "space swimmingpool": "room",
    "space den fireplace": "room",
    "space elevated": "room",
    "space dressingroom": "room",
    "space alcove": "room",
    "space bath": "bathroom",
    "space bath shower": "bathroom",
    "space kitchen": "kitchen",
    "space kitchen kitchenette": "kitchen",
    "space kitchen scullery": "kitchen",
    "space kitchen open": "kitchen",
    "space livingroom": "living_room",
    "space bedroom": "bedroom",
    "space bedroom guest": "bedroom",
    "space dining": "dining_room",
    "space outdoor": "outdoor",
    "space outdoor balcony": "outdoor",
    "space outdoor balcony glazed": "outdoor",
    "space outdoor terrace": "outdoor",
    "space outdoor porch": "outdoor",
    "space outdoor coveredarea": "outdoor",
    "space outdoor garden": "outdoor",
    "space utility laundry": "outdoor",  # utility/laundry room: closest bucket, low volume
    # Fixed bathroom fixtures
    "fixedfurniture toilet": "toilet",
    "fixedfurniture sink": "sink",
    "fixedfurniture doublesink": "sink",
    "fixedfurniture doublesinkright": "sink",
    "fixedfurniture roundsink": "sink",
    "fixedfurniture sidesink": "sink",
    "fixedfurniture bathtub": "bathtub",
    "fixedfurniture shower": "shower",
    "fixedfurniture showerscreen": "shower",
    "fixedfurniture showercab": "shower",
    "fixedfurniture showerscreenroundright": "shower",
    "fixedfurniture showerscreenroundleft": "shower",
    # Storage
    "fixedfurniture closet": "wardrobe",
    "fixedfurniture coatcloset": "wardrobe",
    "fixedfurniture coatrack": "wardrobe",
    "fixedfurniture basecabinet": "cabinet",
    "fixedfurniture wallcabinet": "cabinet",
    "fixedfurniture countertop": "kitchen_counter",
    # Fireplace
    "fixedfurniture chimney": "fireplace",
    "fixedfurniture fireplace": "fireplace",
    "fixedfurniture fireplacecorner": "fireplace",
    "fixedfurniture woodstove": "fireplace",
    # Electrical appliances (kitchen + laundry + heating, all folded together --
    # low individual volume per subtype, not worth 8 separate YOLO classes)
    "fixedfurniture electricalappliance": "electrical_appliance",
    "fixedfurniture electricalappliance refrigerator": "electrical_appliance",
    "fixedfurniture electricalappliance integratedstove": "electrical_appliance",
    "fixedfurniture electricalappliance stove": "electrical_appliance",
    "fixedfurniture electricalappliance dishwasher": "electrical_appliance",
    "fixedfurniture electricalappliance washingmachine": "electrical_appliance",
    "fixedfurniture electricalappliance tumbledryer": "electrical_appliance",
    "fixedfurniture electricalappliance saunastove": "electrical_appliance",
    "fixedfurniture electricalappliance saunastoveround": "electrical_appliance",
    "fixedfurniture electricalappliance highheater": "electrical_appliance",
    "fixedfurniture electricalappliance heater": "electrical_appliance",
    "fixedfurniture electricalappliance fan": "electrical_appliance",
    "fixedfurniture electricalappliance gearound": "electrical_appliance",
    "fixedfurniture electricalappliance spaceforappliance": "electrical_appliance",
    "fixedfurniture electricalappliance spaceforappliance2": "electrical_appliance",
    # Benches (incl. sauna benches -- same physical shape)
    "bench": "bench",
    "fixedfurniture saunabench": "bench",
    "fixedfurniture saunabenchmid": "bench",
    "fixedfurniture saunabenchhigh": "bench",
    "fixedfurniture saunabenchlow": "bench",
    # Misc / unclassifiable fixed furniture
    "fixedfurniture misc": "misc_furniture",
    "fixedfurniture housing": "misc_furniture",
}

# Raw classes that DO match a recognized semantic prefix (so the parser
# attaches geometry to them) but are wrapper/group classes with no real
# furniture meaning of their own -- e.g. "FixedFurnitureSet" groups several
# individual `FixedFurniture X` children, each already labeled correctly by
# the parser's nearest-ancestor resolution. Any geometry landing directly on
# one of these (rather than on a nested, more specific child) is noise.
EXCLUDED_RAW_CLASSES: Set[str] = {
    "fixedfurnitureset",
}

_WHITESPACE_RE = re.compile(r"\s+")

# Prefix -> unified class, used only when the normalized raw class isn't an
# exact match in RAW_TO_UNIFIED above. Order matters: first match wins.
_PREFIX_FALLBACKS = (
    ("wall", "wall"),
    ("space ", "room"),
    ("fixedfurniture electricalappliance", "electrical_appliance"),
    ("fixedfurniture sauna", "bench"),
    ("fixedfurniture", "misc_furniture"),
    ("doors", "door"),
    ("door", "door"),
    ("window", "window"),
    ("column", "column"),
    ("railing", "stairs"),
    ("stairs", "stairs"),
    ("bench", "bench"),
)

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
        excluded (a wrapper/group class) or is unrecognized. Unrecognized
        classes are logged once per run for later taxonomy iteration -- they
        are never silently assigned to `misc_furniture`.
        """
        normalized = cls.normalize_raw_class(raw_class)

        if normalized in EXCLUDED_RAW_CLASSES:
            return None

        if normalized in RAW_TO_UNIFIED:
            return RAW_TO_UNIFIED[normalized]

        for prefix, unified in _PREFIX_FALLBACKS:
            if normalized.startswith(prefix):
                return unified

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
