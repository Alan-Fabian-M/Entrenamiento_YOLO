"""Room contour extraction ("the compass and ruler" of the pipeline).

Ported from the original `SpatialSceneCompiler/scene_compiler.py` prototype.
Instead of reconstructing every wall as an individual segment (fragile against
hand-drawn, non-orthogonal lines), we locate the room as a bounding box.

Two strategies, tried in order:

1. Enclosed-space detection (primary): find the empty regions walled off from
   the page border and take the largest one. This is what actually reads as
   "a room" on a technical/CAD plan, where the largest *edge* contour is the
   whole drawing frame -- not a room. Walls are dilated first so door gaps
   don't leak a room into the corridor.
2. Largest external contour (fallback): the original heuristic, which still
   works for a simple single-room sketch photographed on light paper where the
   room outline *is* the dominant contour and strategy 1 finds nothing framed.

`build_wall_elements()` turns those bounds into four `SceneElement`-shaped wall
dicts (muro_norte/sur/este/oeste), matching the exact style already used in
`Assets/Resources/ScenePresets/preset_*.json`. This means a compiled photo
produces a full walled room (not just floating furniture), consistent with
what the presets already demonstrate in Unity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

# An enclosed region smaller than this fraction of the page is noise (a symbol,
# a stray gap), not a room. Keeps strategy 1 from latching onto tiny pockets.
_MIN_ROOM_AREA_FRACTION = 0.01


@dataclass
class RoomContour:
    origin_px: Tuple[int, int]
    width_px: int
    height_px: int


class RoomExtractor:
    @staticmethod
    def detect_bounds(image_bgr: np.ndarray) -> Optional[RoomContour]:
        return RoomExtractor._detect_enclosed_room(image_bgr) or RoomExtractor._detect_largest_contour(image_bgr)

    @staticmethod
    def _detect_enclosed_room(image_bgr: np.ndarray) -> Optional[RoomContour]:
        """Find the largest empty region fully walled off from the page border.

        This treats a room as negative space enclosed by walls, which is what
        survives on a CAD plan whose outermost edge contour is just the drawing
        frame. Returns None if no interior region qualifies, so the caller can
        fall back to the classic largest-contour heuristic.
        """
        h_img, w_img = image_bgr.shape[:2]

        # Polarity-aware binarization: walls become dark (0) on a light (255)
        # background whether the source is dark ink on paper or bright lines on
        # a dark CAD canvas.
        v_channel = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)[:, :, 2]
        _, binary = cv2.threshold(v_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if cv2.countNonZero(binary) < binary.size // 2:
            binary = 255 - binary

        # Thicken walls so door openings close up and a room stays sealed off
        # from the corridor / neighboring unit.
        walls = cv2.dilate(255 - binary, np.ones((7, 7), np.uint8), iterations=1)
        enclosed_space = 255 - walls

        num_labels, _labels, stats, _centroids = cv2.connectedComponentsWithStats(enclosed_space, connectivity=4)

        min_area = _MIN_ROOM_AREA_FRACTION * w_img * h_img
        best: Optional[Tuple[int, int, int, int]] = None  # (area, x, y, w, h) -> stored as bbox
        best_area = 0
        for label_id in range(1, num_labels):  # skip background label 0
            x, y, w, h, area = stats[label_id]
            touches_border = x <= 1 or y <= 1 or (x + w) >= w_img - 1 or (y + h) >= h_img - 1
            if touches_border or area < min_area:
                continue
            if area > best_area:
                best_area = area
                best = (x, y, w, h)

        if best is None:
            return None

        x, y, w, h = best
        return RoomContour(origin_px=(int(x), int(y)), width_px=int(w), height_px=int(h))

    @staticmethod
    def _detect_largest_contour(image_bgr: np.ndarray) -> Optional[RoomContour]:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 40, 120)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        return RoomContour(origin_px=(x, y), width_px=w, height_px=h)

    @staticmethod
    def build_wall_elements(width_m: float, depth_m: float, height_m: float, wall_thickness_m: float = 0.2,
                             wall_color: str = "#FFFFFF") -> list:
        """Return 4 wall scene_elements (type='muro') framing a width_m x depth_m room,
        in the same shape/convention as Assets/Resources/ScenePresets/preset_*.json.
        """
        half_d = depth_m / 2.0
        material = {"type": "concrete", "color": wall_color}

        def _make_wall(id_str, px, py, pz, sx, sy, sz):
            pos = {"x": px, "y": py, "z": pz}
            rot = {"x": 0, "y": 0, "z": 0}
            scale = {"x": sx, "y": sy, "z": sz}
            return {
                "id": id_str,
                "type": "muro",
                "class_label": "muro",
                "prefab_id": "muro",
                "confidence": 0.85,
                "position": pos,
                "rotation": rot,
                "scale": scale,
                "material": material,
                "transform": {
                    "position": pos,
                    "rotation": rot,
                    "scale": scale,
                },
                "properties": {
                    "color_hex": wall_color,
                    "material_type": "concrete",
                },
            }

        return [
            _make_wall("muro_norte", 0, 0, depth_m, width_m, height_m, wall_thickness_m),
            _make_wall("muro_sur", 0, 0, 0, width_m, height_m, wall_thickness_m),
            _make_wall("muro_este", width_m, 0, half_d, wall_thickness_m, height_m, depth_m),
            _make_wall("muro_oeste", 0, 0, half_d, wall_thickness_m, height_m, depth_m),
        ]
