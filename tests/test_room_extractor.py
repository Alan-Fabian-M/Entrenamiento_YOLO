"""Tests for multi-room detection (`RoomExtractor.detect_rooms`/`build_room_walls`)
and its wiring into `POST /api/v1/compilar-sala`.

Follows the synthetic-image + FastAPI `TestClient` convention already used in
`tests/test_preprocess.py`.
"""

import io

import cv2
import numpy as np
from fastapi.testclient import TestClient

from app.main import app
from app.services.room_extractor import RoomExtractor, RoomShape

client = TestClient(app)


def create_synthetic_two_room_floorplan(width: int = 400, height: int = 300) -> bytes:
    """Two rooms side by side, split by an interior wall with a small door gap
    (well within the 7x7/1-iteration dilation's closing radius of ~6px) that
    `detect_rooms` must seal so the two rooms don't merge into one region.
    """
    img = np.full((height, width, 3), 250, dtype=np.uint8)
    wall_color = (20, 20, 20)
    thickness = 6

    cv2.rectangle(img, (20, 20), (width - 20, height - 20), wall_color, thickness)

    mid_x = width // 2
    cv2.line(img, (mid_x, 20), (mid_x, 130), wall_color, thickness)
    cv2.line(img, (mid_x, 134), (mid_x, height - 20), wall_color, thickness)  # 4px door gap

    success, encoded = cv2.imencode(".png", img)
    assert success
    return encoded.tobytes()


def test_detect_rooms_finds_two_rooms():
    image_bytes = create_synthetic_two_room_floorplan()
    bgr = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)

    rooms = RoomExtractor.detect_rooms(bgr)

    assert len(rooms) == 2
    for room in rooms:
        assert 130 <= room.width_px <= 200
        assert 220 <= room.height_px <= 270
        assert len(room.contour_px) >= 3

    left, right = sorted(rooms, key=lambda r: r.origin_px[0])
    assert left.origin_px[0] < right.origin_px[0]
    # Neither room should swallow the other -- their x-ranges shouldn't overlap.
    assert left.origin_px[0] + left.width_px <= right.origin_px[0] + 5


def test_detect_rooms_returns_empty_for_blank_image():
    blank = np.full((300, 400, 3), 255, dtype=np.uint8)
    assert RoomExtractor.detect_rooms(blank) == []


def test_build_room_walls_ids_namespaced():
    room_a = RoomShape(room_id="room_0", origin_px=(0, 0), width_px=100, height_px=100,
                        contour_px=np.array([[0, 0], [100, 0], [100, 100], [0, 100]]))
    room_b = RoomShape(room_id="room_1", origin_px=(100, 0), width_px=100, height_px=100,
                        contour_px=np.array([[100, 0], [200, 0], [200, 100], [100, 100]]))

    walls_a = RoomExtractor.build_room_walls(room_a, width_m=4.0, depth_m=4.0, height_m=2.5, center_m=(0.0, 0.0))
    walls_b = RoomExtractor.build_room_walls(room_b, width_m=4.0, depth_m=4.0, height_m=2.5, center_m=(4.0, 0.0))

    assert len(walls_a) == 4
    assert len(walls_b) == 4
    ids_a = {w["id"] for w in walls_a}
    ids_b = {w["id"] for w in walls_b}
    assert ids_a.isdisjoint(ids_b)
    assert all(rot == {"x": 0, "y": 0, "z": 0} for w in walls_a for rot in [w["rotation"]])


def test_compile_sala_multi_room_end_to_end():
    image_bytes = create_synthetic_two_room_floorplan()
    files = {"file": ("plano_dos_cuartos.png", io.BytesIO(image_bytes), "image/png")}

    response = client.post("/api/v1/compilar-sala", files=files)
    assert response.status_code == 200
    data = response.json()

    assert len(data["rooms"]) == 2
    assert data["metadata"]["room_count"] == 2

    # room_info (singular, backward-compat) must keep its original shape.
    assert "room_type" in data["room_info"]
    assert "dimensions" in data["room_info"]

    wall_elements = [el for el in data["scene_elements"] if el["type"] == "muro"]
    assert len(wall_elements) == 4 * data["metadata"]["room_count"]
    wall_ids = {el["id"] for el in wall_elements}
    assert len(wall_ids) == len(wall_elements)  # no id collisions across rooms
