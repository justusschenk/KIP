"""Tests for kip/data/yolo_to_coco.py."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from kip import CLASS_NAMES
from kip.data.yolo_to_coco import convert_split, parse_yolo_seg_line


# ---------------------------------------------------------------------------
# parse_yolo_seg_line
# ---------------------------------------------------------------------------

def test_parse_line_basic():
    line = "0 0.1 0.1 0.9 0.1 0.9 0.9 0.1 0.9"
    cls, pts = parse_yolo_seg_line(line, w=100, h=100)
    assert cls == 0
    assert pts.shape == (4, 2)
    # denormalised
    assert abs(pts[0, 0] - 10.0) < 1e-6
    assert abs(pts[0, 1] - 10.0) < 1e-6


def test_parse_line_class_3():
    line = "3 0.0 0.0 1.0 0.0 1.0 1.0 0.0 1.0"
    cls, pts = parse_yolo_seg_line(line, w=640, h=480)
    assert cls == 3
    assert pts.shape == (4, 2)


def test_parse_line_too_few_tokens():
    with pytest.raises(ValueError):
        parse_yolo_seg_line("0 0.5 0.5", w=100, h=100)


def test_parse_line_odd_coords():
    with pytest.raises(ValueError):
        parse_yolo_seg_line("0 0.1 0.2 0.3", w=100, h=100)


# ---------------------------------------------------------------------------
# convert_split
# ---------------------------------------------------------------------------

def test_convert_normal_split(yolo_seg_dataset, tmp_path):
    root, W, H = yolo_seg_dataset
    out_json = tmp_path / "test.json"
    coco = convert_split(
        images_dir=root / "images",
        labels_dir=root / "labels",
        class_names=CLASS_NAMES,
        out_json=out_json,
    )

    assert out_json.exists()
    assert len(coco["images"]) == 4  # all 4 images
    assert len(coco["categories"]) == len(CLASS_NAMES)

    # img_001 has 2 valid annotations
    img1 = next(im for im in coco["images"] if "img_001" in im["file_name"])
    anns_for_img1 = [a for a in coco["annotations"] if a["image_id"] == img1["id"]]
    assert len(anns_for_img1) == 2

    # img_002 (no label file) -> zero annotations
    img2 = next(im for im in coco["images"] if "img_002" in im["file_name"])
    anns_for_img2 = [a for a in coco["annotations"] if a["image_id"] == img2["id"]]
    assert len(anns_for_img2) == 0

    # img_003 (empty label file) -> zero annotations
    img3 = next(im for im in coco["images"] if "img_003" in im["file_name"])
    anns_for_img3 = [a for a in coco["annotations"] if a["image_id"] == img3["id"]]
    assert len(anns_for_img3) == 0

    # img_004 (<3-point polygon) -> zero annotations
    img4 = next(im for im in coco["images"] if "img_004" in im["file_name"])
    anns_for_img4 = [a for a in coco["annotations"] if a["image_id"] == img4["id"]]
    assert len(anns_for_img4) == 0


def test_convert_annotation_fields(yolo_seg_dataset, tmp_path):
    root, W, H = yolo_seg_dataset
    coco = convert_split(
        images_dir=root / "images",
        labels_dir=root / "labels",
        class_names=CLASS_NAMES,
        out_json=tmp_path / "out.json",
    )
    for ann in coco["annotations"]:
        assert "id" in ann
        assert "image_id" in ann
        assert "category_id" in ann
        assert "segmentation" in ann
        assert "bbox" in ann
        assert "area" in ann
        assert ann["iscrowd"] == 0
        assert ann["area"] > 0
        assert len(ann["bbox"]) == 4


def test_convert_image_dims(yolo_seg_dataset, tmp_path):
    root, W, H = yolo_seg_dataset
    coco = convert_split(
        images_dir=root / "images",
        labels_dir=root / "labels",
        class_names=CLASS_NAMES,
        out_json=tmp_path / "out.json",
    )
    for img in coco["images"]:
        assert img["width"] == W
        assert img["height"] == H


def test_convert_start_ids(yolo_seg_dataset, tmp_path):
    """start_ids should offset image and annotation IDs."""
    root, W, H = yolo_seg_dataset
    coco = convert_split(
        images_dir=root / "images",
        labels_dir=root / "labels",
        class_names=CLASS_NAMES,
        out_json=tmp_path / "out.json",
        start_ids=(100, 1000),
    )
    assert coco["images"][0]["id"] >= 100
    if coco["annotations"]:
        assert coco["annotations"][0]["id"] >= 1000


def test_convert_validates_with_pycocotools(yolo_seg_dataset, tmp_path):
    """Output must load cleanly with pycocotools.coco.COCO."""
    from pycocotools.coco import COCO

    root, W, H = yolo_seg_dataset
    out_json = tmp_path / "val_coco.json"
    convert_split(
        images_dir=root / "images",
        labels_dir=root / "labels",
        class_names=CLASS_NAMES,
        out_json=out_json,
    )
    coco_api = COCO(str(out_json))
    assert len(coco_api.imgs) == 4


def test_convert_polygon_denormalised(yolo_seg_dataset, tmp_path):
    """Polygon coordinates must be in absolute pixels, not normalised."""
    root, W, H = yolo_seg_dataset
    coco = convert_split(
        images_dir=root / "images",
        labels_dir=root / "labels",
        class_names=CLASS_NAMES,
        out_json=tmp_path / "out.json",
    )
    for ann in coco["annotations"]:
        seg = ann["segmentation"][0]
        xs = seg[0::2]
        ys = seg[1::2]
        # Absolute coords should exceed 1.0 for non-trivially small images
        assert max(xs) > 1.0 or max(ys) > 1.0
