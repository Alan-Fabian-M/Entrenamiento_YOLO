"""Tests for the CubiCasa5K -> YOLOv8-OBB conversion pipeline.

Uses synthetic in-memory SVG/image fixtures (no real CubiCasa5K dataset
required), mirroring the synthetic-floorplan approach in test_preprocess.py.
"""

import numpy as np
import cv2
import pytest
from lxml import etree

from training.cubicasa.svg_transform import SvgTransform, SvgTransformError
from training.cubicasa.class_mapping import ClassMapper
from training.cubicasa.svg_parser import CubiCasaSvgParser
from training.cubicasa.obb_converter import ObbConverter, ObbConversionError
from training.cubicasa.dataset_builder import BuildConfig, YoloObbDatasetBuilder


# --- svg_transform.py -------------------------------------------------------

def test_transform_composition_translate_then_scale():
    matrix = SvgTransform.parse_transform_attr("translate(10,20) scale(2)")
    result = SvgTransform.apply_to_points(matrix, np.array([[1.0, 1.0]]))
    # scale(2) applied first (local), then translate: (1*2+10, 1*2+20)
    np.testing.assert_allclose(result, [[12.0, 22.0]])


def test_transform_rotate_matches_manual_rotation_matrix():
    matrix = SvgTransform.parse_transform_attr("rotate(90)")
    result = SvgTransform.apply_to_points(matrix, np.array([[1.0, 0.0]]))
    np.testing.assert_allclose(result, [[0.0, 1.0]], atol=1e-9)


def test_transform_translate_then_rotate_order():
    matrix = SvgTransform.parse_transform_attr("translate(10,0) rotate(90)")
    result = SvgTransform.apply_to_points(matrix, np.array([[1.0, 0.0]]))
    np.testing.assert_allclose(result, [[10.0, 1.0]], atol=1e-9)


def test_transform_nested_group_composition():
    parent = SvgTransform.parse_transform_attr("translate(10,10)")
    child = SvgTransform.parse_transform_attr("scale(2)")
    composed = SvgTransform.compose(parent, child)
    result = SvgTransform.apply_to_points(composed, np.array([[1.0, 1.0]]))
    np.testing.assert_allclose(result, [[12.0, 12.0]])


def test_transform_empty_attr_is_identity():
    matrix = SvgTransform.parse_transform_attr(None)
    np.testing.assert_array_equal(matrix, SvgTransform.IDENTITY)


def test_transform_invalid_function_raises():
    with pytest.raises(SvgTransformError):
        SvgTransform.parse_transform_attr("bogus(1,2)")


# --- class_mapping.py --------------------------------------------------------

def test_class_mapper_normalizes_case_and_whitespace():
    ClassMapper.reset_unrecognized_classes()
    assert ClassMapper.map_to_unified("  SOFA  ") == "sofa"
    assert ClassMapper.map_to_unified("Kitchen   Cabinet") == "cabinet"


def test_class_mapper_returns_none_for_excluded_class():
    ClassMapper.reset_unrecognized_classes()
    assert ClassMapper.map_to_unified("Wall") is None
    assert "wall" not in ClassMapper.get_unrecognized_classes()


def test_class_mapper_reports_unrecognized_raw_class():
    ClassMapper.reset_unrecognized_classes()
    result = ClassMapper.map_to_unified("SomeUnknownIconType")
    assert result is None
    assert ClassMapper.normalize_raw_class("SomeUnknownIconType") in ClassMapper.get_unrecognized_classes()


def test_class_mapper_class_id_matches_declared_order():
    from training.cubicasa.class_mapping import YOLO_CLASSES
    assert ClassMapper.class_id("door") == 0
    assert ClassMapper.class_id("window") == 1
    assert YOLO_CLASSES[ClassMapper.class_id("sofa")] == "sofa"


# --- svg_parser.py ------------------------------------------------------------

def _build_sample_svg() -> etree._Element:
    svg = etree.Element("svg", nsmap=None)
    svg.set("viewBox", "0 0 100 100")

    outer_group = etree.SubElement(svg, "g")
    outer_group.set("transform", "translate(10,10)")

    inner_group = etree.SubElement(outer_group, "g")
    inner_group.set("transform", "scale(2)")

    sofa = etree.SubElement(inner_group, "polygon")
    sofa.set("class", "Sofa")
    sofa.set("points", "0,0 5,0 5,5 0,5")

    wall = etree.SubElement(svg, "polygon")
    wall.set("class", "Wall")
    wall.set("points", "0,0 100,0 100,2 0,2")

    door = etree.SubElement(svg, "polygon")
    door.set("class", "Door 1-panel")
    door.set("transform", "translate(50,50) rotate(45)")
    door.set("points", "0,0 10,0 10,4 0,4")

    degenerate = etree.SubElement(svg, "polygon")
    degenerate.set("class", "Sink")
    degenerate.set("points", "0,0 1,1 2,2")  # collinear -> zero area

    straight_path = etree.SubElement(svg, "path")
    straight_path.set("class", "Table")
    straight_path.set("d", "M 20,20 L 30,20 L 30,30 L 20,30 Z")

    curved_path = etree.SubElement(svg, "path")
    curved_path.set("class", "Chair")
    curved_path.set("d", "M 40,40 C 45,45 55,45 60,40")

    return svg


def test_svg_parser_excludes_nothing_at_walk_level_but_applies_viewbox_scale():
    svg_root = _build_sample_svg()
    image_shape = (200, 200, 3)  # 2x viewBox -> scale factor 2
    annotations = CubiCasaSvgParser.walk_and_collect(svg_root, image_shape)

    classes_found = {a.raw_class for a in annotations}
    assert "Sofa" in classes_found
    assert "Wall" in classes_found  # walk_and_collect doesn't filter; class_mapping does
    assert "Chair" not in classes_found  # curved path unsupported, dropped


def test_svg_parser_applies_viewbox_scale_to_image_dimensions():
    svg_root = _build_sample_svg()
    image_shape = (200, 200, 3)
    annotations = CubiCasaSvgParser.walk_and_collect(svg_root, image_shape)

    sofa_ann = next(a for a in annotations if a.raw_class == "Sofa")
    # local (0,0)->scale(2)->(0,0); translate(10,10)->(10,10); viewbox scale x2 -> (20,20)
    np.testing.assert_allclose(sofa_ann.points_px[0], [20.0, 20.0])
    # local (5,5)->scale(2)->(10,10); translate(10,10)->(20,20); viewbox scale x2 -> (40,40)
    np.testing.assert_allclose(sofa_ann.points_px[2], [40.0, 40.0])


def test_svg_parser_straight_path_is_extracted():
    svg_root = _build_sample_svg()
    image_shape = (100, 100, 3)
    annotations = CubiCasaSvgParser.walk_and_collect(svg_root, image_shape)
    table_ann = next(a for a in annotations if a.raw_class == "Table")
    assert table_ann.points_px.shape[0] >= 4


# --- obb_converter.py ---------------------------------------------------------

def test_obb_converter_filters_degenerate_polygon():
    collinear = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    assert ObbConverter.is_valid_polygon(collinear) is False
    assert ObbConverter.polygon_to_obb(collinear) is None


def test_obb_converter_produces_4_corners_within_0_1_after_normalization():
    square = np.array([[10.0, 10.0], [50.0, 10.0], [50.0, 50.0], [10.0, 50.0]])
    corners_px = ObbConverter.polygon_to_obb(square)
    assert corners_px.shape == (4, 2)

    normalized = ObbConverter.normalize_corners(corners_px, img_w=100, img_h=100)
    assert normalized.shape == (4, 2)
    assert np.all(normalized >= 0.0) and np.all(normalized <= 1.0)


def test_obb_converter_corner_order_is_deterministic_across_calls():
    rotated = np.array([[10.0, 0.0], [20.0, 10.0], [10.0, 20.0], [0.0, 10.0]])
    first = ObbConverter.polygon_to_obb(rotated)
    second = ObbConverter.polygon_to_obb(rotated)
    np.testing.assert_allclose(first, second)


def test_obb_converter_rejects_invalid_image_dimensions():
    corners = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    with pytest.raises(ObbConversionError):
        ObbConverter.normalize_corners(corners, img_w=0, img_h=100)


def test_obb_converter_to_yolo_obb_line_format():
    corners_norm = np.array([[0.1, 0.1], [0.2, 0.1], [0.2, 0.2], [0.1, 0.2]])
    line = ObbConverter.to_yolo_obb_line(3, corners_norm)
    parts = line.split()
    assert parts[0] == "3"
    assert len(parts) == 9


# --- dataset_builder.py --------------------------------------------------------

def _write_synthetic_sample(sample_dir, svg_root, image_size=(100, 100)):
    sample_dir.mkdir(parents=True, exist_ok=True)
    image = np.full((image_size[1], image_size[0], 3), 255, dtype=np.uint8)
    cv2.imwrite(str(sample_dir / "F1_scaled.png"), image)
    tree = etree.ElementTree(svg_root)
    tree.write(str(sample_dir / "model.svg"))


def test_dataset_builder_writes_expected_directory_structure(tmp_path):
    ClassMapper.reset_unrecognized_classes()
    cubicasa_root = tmp_path / "cubicasa5k"
    sample_relpath = "high_quality/sample_001"
    _write_synthetic_sample(cubicasa_root / sample_relpath, _build_sample_svg())

    train_split = tmp_path / "train.txt"
    train_split.write_text(sample_relpath + "\n")
    val_split = tmp_path / "val.txt"
    val_split.write_text("")
    test_split = tmp_path / "test.txt"
    test_split.write_text("")

    output_root = tmp_path / "yolo_dataset"
    config = BuildConfig(
        cubicasa_root=cubicasa_root,
        output_root=output_root,
        splits={"train": train_split, "val": val_split, "test": test_split},
    )

    report = YoloObbDatasetBuilder.build(config)

    sample_id = "high_quality_sample_001"
    assert (output_root / "images" / "train" / f"{sample_id}.png").exists()
    assert (output_root / "labels" / "train" / f"{sample_id}.txt").exists()
    assert (output_root / "data.yaml").exists()
    assert report.total_samples == 1
    assert report.skipped_samples == 0


def test_data_yaml_contains_all_class_names_in_order(tmp_path):
    from training.cubicasa.class_mapping import YOLO_CLASSES
    YoloObbDatasetBuilder.write_data_yaml(tmp_path, YOLO_CLASSES)
    content = (tmp_path / "data.yaml").read_text()
    for i, name in enumerate(YOLO_CLASSES):
        assert f"{i}: {name}" in content
