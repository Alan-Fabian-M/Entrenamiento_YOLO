"""Room contour extraction ("the compass and ruler" of the pipeline).

Ported from the original `SpatialSceneCompiler/scene_compiler.py` prototype.
Instead of reconstructing every wall as an individual segment (fragile against
hand-drawn, non-orthogonal lines), we locate each room as a bounding box.

Two strategies, tried in order:

1. Enclosed-space detection (primary, `detect_rooms`): find every empty
   region walled off from the page border -- not just the largest -- so a
   multi-room CAD plan (the common case) yields one RoomShape per real room
   instead of being flattened into a single fake bounding box. This is what
   actually reads as "a room" on a technical/CAD plan, where the largest
   *edge* contour is the whole drawing frame, not a room. Walls are dilated
   first so door gaps don't leak a room into the corridor. Rooms are treated
   as axis-aligned boxes for wall generation (`build_room_walls`) -- CubiCasa
   plans are practically always axis-aligned per room -- but each RoomShape
   also keeps its real (simplified) contour for accurate point-in-polygon
   furniture assignment, done by the caller (`SceneCompiler`).
2. Largest external contour (fallback, `_detect_largest_contour`): the
   original single-room heuristic, which still works for a simple sketch
   photographed on light paper where the room outline *is* the dominant
   contour and strategy 1 finds nothing framed.

`build_wall_elements()` (single room, world-origin-anchored) and
`build_room_walls()` (one room among several, centered on its own position)
turn bounds into `SceneElement`-shaped wall dicts (muro_norte/sur/este/oeste),
matching the exact style already used in
`Assets/Resources/ScenePresets/preset_*.json`. This means a compiled photo
produces a full walled room -- or several -- (not just floating furniture),
consistent with what the presets already demonstrate in Unity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

# An enclosed region smaller than this fraction of the page is noise (a symbol,
# a stray gap), not a room. Keeps strategy 1 from latching onto tiny pockets.
# Tuned lower than a single-room heuristic would need: in a multi-room plan a
# real bathroom/closet is legitimately much smaller relative to the whole
# page than "the room" is when there's only one.
_MIN_ROOM_AREA_FRACTION = 0.004


@dataclass
class RoomContour:
    origin_px: Tuple[int, int]
    width_px: int
    height_px: int


@dataclass
class RoomShape:
    """One detected enclosed room, kept distinct from `RoomContour` because it
    also carries its real contour (for point-in-polygon furniture assignment)
    and a reserved rotation field, not just a bbox.
    """

    room_id: str
    origin_px: Tuple[int, int]
    width_px: int
    height_px: int
    contour_px: np.ndarray  # Nx2 polygon points, simplified via approxPolyDP
    rotation_deg: float = 0.0  # reserved for future use; walls stay axis-aligned in v1


class RoomExtractor:
    @staticmethod
    def detect_bounds(image_bgr: np.ndarray) -> Optional[RoomContour]:
        return RoomExtractor._detect_enclosed_room(image_bgr) or RoomExtractor._detect_largest_contour(image_bgr)

    @staticmethod
    def detect_rooms(image_bgr: np.ndarray) -> List[RoomShape]:
        """Find every empty region fully walled off from the page border.

        This treats each room as negative space enclosed by walls, which is
        what survives on a CAD plan whose outermost edge contour is just the
        drawing frame. Returns one RoomShape per qualifying enclosed region,
        largest first. Returns [] if no interior region qualifies, so the
        caller can fall back to the classic largest-contour heuristic (a
        single room spanning the whole detected outline).

        Note: `connectedComponentsWithStats` assigns each pixel to exactly one
        label, so the returned rooms are guaranteed disjoint -- the failure
        mode to watch for is under/over-dilation merging two rooms into one
        component (door gap not sealed) or splitting one room into two
        (wall line not fully closed), not overlap.
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

        num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(enclosed_space, connectivity=4)

        min_area = _MIN_ROOM_AREA_FRACTION * w_img * h_img
        candidates: List[Tuple[int, int, int, int, int, int]] = []  # (area, x, y, w, h, label_id)
        for label_id in range(1, num_labels):  # skip background label 0
            x, y, w, h, area = stats[label_id]
            touches_border = x <= 1 or y <= 1 or (x + w) >= w_img - 1 or (y + h) >= h_img - 1
            if touches_border or area < min_area:
                continue
            candidates.append((area, x, y, w, h, label_id))

        candidates.sort(key=lambda c: c[0], reverse=True)

        rooms: List[RoomShape] = []
        for i, (_area, x, y, w, h, label_id) in enumerate(candidates):
            label_mask = (labels == label_id).astype(np.uint8) * 255
            contour_px = _largest_contour_points(label_mask)
            rooms.append(
                RoomShape(
                    room_id=f"room_{i}",
                    origin_px=(int(x), int(y)),
                    width_px=int(w),
                    height_px=int(h),
                    contour_px=contour_px,
                )
            )
        return rooms

    @staticmethod
    def _detect_enclosed_room(image_bgr: np.ndarray) -> Optional[RoomContour]:
        """Single-room convenience wrapper over `detect_rooms` (largest room),
        kept for the existing single-room fallback path.
        """
        rooms = RoomExtractor.detect_rooms(image_bgr)
        if not rooms:
            return None
        largest = rooms[0]
        return RoomContour(origin_px=largest.origin_px, width_px=largest.width_px, height_px=largest.height_px)

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

    @staticmethod
    def build_room_walls(room: RoomShape, width_m: float, depth_m: float, height_m: float,
                          center_m: Tuple[float, float], wall_thickness_m: float = 0.2,
                          wall_color: str = "#FFFFFF") -> list:
        """Same wall shape/convention as `build_wall_elements`, but for one room
        among several: walls are centered on `center_m` (not anchored at world
        origin) and ids are namespaced with `room.room_id` so multiple rooms'
        walls don't collide in the flat `scene_elements` list.

        `rotation` stays {0,0,0} for every wall (v1 scope: rooms are treated as
        axis-aligned boxes; `room.rotation_deg` is reserved for later use once
        the Unity wall prefab's rotation pivot is verified).
        """
        cx, cz = center_m
        half_w, half_d = width_m / 2.0, depth_m / 2.0
        material = {"type": "concrete", "color": wall_color}

        def _make_wall(suffix, px, py, pz, sx, sy, sz):
            pos = {"x": px, "y": py, "z": pz}
            rot = {"x": 0, "y": 0, "z": 0}
            scale = {"x": sx, "y": sy, "z": sz}
            return {
                "id": f"muro_{suffix}_{room.room_id}",
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
            _make_wall("norte", cx, 0, cz + half_d, width_m, height_m, wall_thickness_m),
            _make_wall("sur", cx, 0, cz - half_d, width_m, height_m, wall_thickness_m),
            _make_wall("este", cx + half_w, 0, cz, wall_thickness_m, height_m, depth_m),
            _make_wall("oeste", cx - half_w, 0, cz, wall_thickness_m, height_m, depth_m),
        ]


def _largest_contour_points(label_mask: np.ndarray) -> np.ndarray:
    """Extract the (simplified) outer contour of a single connected-component
    mask as an Nx2 array of (x, y) pixel points, for shapely point-in-polygon
    furniture assignment. Falls back to the mask's bounding box if, for some
    reason, no contour is found (shouldn't happen for a non-empty mask).
    """
    contours, _ = cv2.findContours(label_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        ys, xs = np.where(label_mask > 0)
        x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
        return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float64)

    largest = max(contours, key=cv2.contourArea)
    epsilon = 0.01 * cv2.arcLength(largest, True)
    simplified = cv2.approxPolyDP(largest, epsilon, True)
    return simplified.reshape(-1, 2).astype(np.float64)
