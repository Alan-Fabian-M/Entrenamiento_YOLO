"""Tests for the furniture detector's class mapping, dedupe and inverted pass."""

import numpy as np

from app.services.furniture_detector import (
    CLASS_TO_UNITY_TYPE,
    DetectedFurniture,
    FurnitureDetector,
    suppress_overlaps,
)


def _det(unity_type, conf, center, size):
    return DetectedFurniture(unity_type, conf, center, size, 0.0, "#000000")


def test_unified_model_classes_map_to_unity_types():
    assert CLASS_TO_UNITY_TYPE["bed"] == "cama"
    assert CLASS_TO_UNITY_TYPE["refrigerator"] == "refrigerador"
    assert CLASS_TO_UNITY_TYPE["gas_stove"] == "estufa"
    assert CLASS_TO_UNITY_TYPE["shower"] == "ducha"


def test_structural_and_area_classes_are_not_mapped():
    for name in ("wall", "room", "bathroom", "kitchen", "living_room", "bedroom", "outdoor"):
        assert name not in CLASS_TO_UNITY_TYPE


def test_suppress_overlaps_keeps_most_confident_of_same_object():
    sofa = _det("sofa", 0.40, (100, 100), (80, 160))
    cama_same_spot = _det("cama", 0.34, (101, 99), (80, 150))
    cama_elsewhere = _det("cama", 0.29, (500, 250), (150, 170))
    kept = suppress_overlaps([cama_same_spot, sofa, cama_elsewhere])
    assert [d.unity_type for d in kept] == ["sofa", "cama"]
    assert kept[1].center_px == (500, 250)


class _FakeBox:
    def __init__(self, cls_id, conf):
        self.cls = [cls_id]
        self.conf = [conf]


class _FakeMask:
    def __init__(self, xy):
        self.xy = [np.array(xy, dtype=np.float32)]


class _FakeResult:
    names = {0: "bed", 1: "door"}

    def __init__(self, cls_id):
        self.boxes = [_FakeBox(cls_id, 0.9)]
        off = 0 if cls_id == 0 else 200  # keep the two classes apart (no overlap)
        self.masks = [_FakeMask([(10 + off, 10), (60 + off, 10), (60 + off, 60), (10 + off, 60)])]
        self.obb = None


class _FakeModel:
    """Records the brightness of every image it is asked to predict on."""

    names = {0: "bed", 1: "door", 2: "table", 3: "sofa", 4: "chair", 5: "refrigerator", 6: "gas_stove"}

    def __init__(self):
        self.brightness = []

    def __call__(self, image, **kwargs):
        self.brightness.append(float(image.mean()))
        return [_FakeResult(0 if image.mean() < 127 else 1)]


def test_light_plans_run_an_inverted_pass_for_furniture_only(monkeypatch):
    model = _FakeModel()
    monkeypatch.setattr(FurnitureDetector, "_get_model", classmethod(lambda cls: model))
    white_plan = np.full((100, 100, 3), 255, dtype=np.uint8)

    result = FurnitureDetector.detect(white_plan)

    assert len(model.brightness) == 2  # original + inverted
    assert sorted(model.brightness) == [0.0, 255.0]
    # Inverted (dark) pass reports "bed" -> kept; original pass reports "door" -> kept.
    assert sorted(d.unity_type for d in result) == ["cama", "puerta"]
