"""Scale detection via OCR ("the reader" of the pipeline).

Ported from `SpatialSceneCompiler/scene_compiler.py`. Supports both PATH-based
and explicit-path Tesseract configurations. On Windows, automatically falls back
to the standard installation path `C:\\Program Files\\Tesseract-OCR\\tesseract.exe`
if the binary is not found on PATH. On Linux/macOS, install via the system
package manager (e.g. `sudo apt install tesseract-ocr`).
"""

from __future__ import annotations

import os
import re
from collections import Counter
from dataclasses import dataclass

import cv2
import numpy as np
import pytesseract

# Windows fallback: point pytesseract at the standard Tesseract install path
# if the binary is not already discoverable on PATH.
_WINDOWS_TESSERACT = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.name == "nt" and os.path.isfile(_WINDOWS_TESSERACT):
    pytesseract.pytesseract.tesseract_cmd = _WINDOWS_TESSERACT

_DIMENSION_PATTERN = re.compile(r"(\d+[.,]?\d*)\s*(m|cm|mts?)\b", re.IGNORECASE)

# CAD floor plans (e.g. exports from AutoCAD/FloorPlanCAD) commonly label
# dimension lines with bare millimeter values and no unit suffix (e.g. "2200",
# "1800"). Only used as a fallback when no explicit-unit match is found, and
# restricted to a plausible wall/room-segment range (0.3m-20m in mm) to avoid
# picking up unrelated numbers (page labels, dates, ids).
_BARE_MM_PATTERN = re.compile(r"\b(\d{3,5})\b")
_BARE_MM_MIN, _BARE_MM_MAX = 100, 20000

# Separate, more common architectural convention: outer dimension lines
# labeled as bare decimal meters with no unit at all (e.g. "5.00", "6.00").
# Tesseract frequently drops the "." on small text ("5.00" -> "500"), which
# is indistinguishable at the regex level from a genuine bare-mm CAD callout
# once the text has already been mangled. Disambiguate by plausibility
# instead: a 3-digit value (100-999) read as millimeters is always under 1m
# -- implausible as the scale reference for an entire room/floor plan --
# while read as meters-with-a-lost-decimal-point (value/100) it lands in
# 1.00m-9.99m, a normal room width. 4-5 digit values keep the plain
# millimeter reading (a real "5.00" can't OCR-mangle into 4-5 digits).
_AMBIGUOUS_BARE_MIN, _AMBIGUOUS_BARE_MAX = 100, 999


@dataclass
class ScaleResult:
    pixels_per_meter: float
    confidence: float
    source: str


class ScaleDetector:
    @staticmethod
    def _binarize_for_ocr(image_bgr: np.ndarray) -> np.ndarray:
        """Binarize for Tesseract, handling both dark ink on light paper (the
        common hand-photographed sketch) and bright/colored lines on a dark
        background (CAD exports like FloorPlanCAD renders). A fixed grayscale
        threshold fails on the latter: a saturated color (e.g. pure green)
        can land right at the cutoff and get lost to antialiasing.

        Using the HSV V (brightness) channel instead of grayscale luminance
        means a bright colored line on black is treated the same as dark ink
        on white -- both are "far from background" in V. Otsu then finds the
        split point automatically, and we normalize the polarity afterwards
        (Tesseract reads best with dark text on a light background) by
        checking which side of the split is the majority (the background).
        """
        v_channel = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)[:, :, 2]
        _, binary = cv2.threshold(v_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Otsu's THRESH_BINARY maps the brighter side to 255. If most pixels
        # are 255, the background is bright (paper) and text is already dark
        # (0) -- correct polarity. If most pixels are 0, the background is
        # dark (CAD canvas) and the bright lines got mapped to 255 -- invert
        # so the background becomes light and the lines/text become dark.
        if cv2.countNonZero(binary) < binary.size // 2:
            binary = 255 - binary

        return ScaleDetector._strip_drawing_strokes(binary)

    @staticmethod
    def _strip_drawing_strokes(binary: np.ndarray) -> np.ndarray:
        """Drop wall/door-arc line art, keeping only character-sized ink blobs.

        A technical floor plan mixes text with long straight walls and door
        swing arcs -- to Tesseract's layout analysis, those strokes look like
        text fragments and derail it even though the digits are individually
        crisp (verified: OCR reads a tight crop of a single dimension label
        perfectly, but fails on the full page). Connected-component filtering
        by size discards anything too tall/wide/large to be a single glyph.
        """
        ink = 255 - binary  # foreground=255 for connectedComponents
        _num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(ink, connectivity=8)

        mask = np.zeros_like(ink)
        for label_id, (x, y, w, h, area) in enumerate(stats):
            if label_id == 0:  # background component
                continue
            if 6 <= h <= 40 and w <= 45 and area >= 6:
                mask[labels == label_id] = 255

        return 255 - mask

    @staticmethod
    def detect(image_bgr: np.ndarray, room_width_px: float) -> ScaleResult:
        """Look for a number+unit ('4.20m', '420cm') written on the sketch and
        cross it with the room's pixel width to get pixels_per_meter.
        """
        binary = ScaleDetector._binarize_for_ocr(image_bgr)

        try:
            text = pytesseract.image_to_string(binary, config="--psm 11")
        except pytesseract.TesseractNotFoundError:
            return ScaleResult(
                pixels_per_meter=0.0,
                confidence=0.0,
                source="tesseract binary not found on PATH -- install it to enable OCR scale detection",
            )

        if not room_width_px:
            return ScaleResult(pixels_per_meter=0.0, confidence=0.0, source="no room contour to scale against")

        matches = _DIMENSION_PATTERN.findall(text)
        if matches:
            value_str, unit = matches[0]
            value = float(value_str.replace(",", "."))
            meters = value if unit.lower().startswith("m") and unit.lower() != "cm" else value / 100.0

            if meters <= 0:
                return ScaleResult(pixels_per_meter=0.0, confidence=0.0, source="invalid dimension value")

            pixels_per_meter = room_width_px / meters
            return ScaleResult(
                pixels_per_meter=round(pixels_per_meter, 2),
                confidence=0.7,
                source=f"OCR:{value_str}{unit} vs room_width:{room_width_px}px",
            )

        # Fallback: bare millimeter callouts typical of CAD floor plan exports
        # (e.g. "2200", "1800") with no unit suffix. Pick the most frequent
        # value in range, since real dimension callouts tend to repeat across
        # a plan (e.g. symmetric rooms) while stray numbers don't.
        bare_values = [
            int(v) for v in _BARE_MM_PATTERN.findall(text) if _BARE_MM_MIN <= int(v) <= _BARE_MM_MAX
        ]
        if not bare_values:
            return ScaleResult(
                pixels_per_meter=0.0,
                confidence=0.0,
                source="no readable dimension text found in the sketch",
            )

        most_common, _count = Counter(bare_values).most_common(1)[0]

        if _AMBIGUOUS_BARE_MIN <= most_common <= _AMBIGUOUS_BARE_MAX:
            meters = most_common / 100.0
            source_note = f"{most_common} (sin punto decimal, asumido {meters:.2f}m)"
        else:
            meters = most_common / 1000.0
            source_note = f"{most_common}mm (sin unidad, asumido mm)"

        pixels_per_meter = room_width_px / meters
        return ScaleResult(
            pixels_per_meter=round(pixels_per_meter, 2),
            confidence=0.5,
            source=f"OCR:{source_note} vs room_width:{room_width_px}px",
        )
