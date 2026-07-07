"""Tests for kip/metrics/anomaly.py and kip/metrics/detection.py."""
from __future__ import annotations

import numpy as np
import pytest

from kip.metrics.anomaly import (
    aupro,
    best_f1,
    dice_iou,
    image_auroc,
    pixel_auroc,
)


# ---------------------------------------------------------------------------
# image_auroc
# ---------------------------------------------------------------------------

def test_image_auroc_perfect():
    labels = [0, 0, 1, 1]
    scores = [0.1, 0.2, 0.8, 0.9]
    assert image_auroc(labels, scores) == pytest.approx(1.0)


def test_image_auroc_random():
    labels = [0, 0, 1, 1]
    scores = [0.5, 0.5, 0.5, 0.5]
    val = image_auroc(labels, scores)
    assert val is not None
    assert 0.0 <= val <= 1.0


def test_image_auroc_single_class_returns_none():
    # All same label -> None
    assert image_auroc([1, 1, 1], [0.1, 0.5, 0.9]) is None
    assert image_auroc([0, 0, 0], [0.1, 0.5, 0.9]) is None


def test_image_auroc_numpy_arrays():
    labels = np.array([0, 1])
    scores = np.array([0.2, 0.8])
    assert image_auroc(labels, scores) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# pixel_auroc
# ---------------------------------------------------------------------------

def test_pixel_auroc_perfect():
    gt = np.array([[0, 0], [1, 1]])
    amap = np.array([[0.1, 0.1], [0.9, 0.9]], dtype=np.float32)
    assert pixel_auroc(gt, amap) == pytest.approx(1.0)


def test_pixel_auroc_single_class_returns_none():
    gt = np.zeros((4, 4), dtype=np.uint8)
    amap = np.random.rand(4, 4).astype(np.float32)
    assert pixel_auroc(gt, amap) is None


def test_pixel_auroc_valid_range():
    rng = np.random.default_rng(42)
    gt = (rng.random((10, 10)) > 0.5).astype(np.uint8)
    amap = rng.random((10, 10)).astype(np.float32)
    val = pixel_auroc(gt, amap)
    if val is not None:
        assert 0.0 <= val <= 1.0


# ---------------------------------------------------------------------------
# aupro
# ---------------------------------------------------------------------------

def test_aupro_returns_none_no_defects():
    gt_masks = [np.zeros((16, 16), dtype=np.uint8)]
    amaps = [np.random.rand(16, 16).astype(np.float32)]
    assert aupro(gt_masks, amaps) is None


def test_aupro_perfect():
    """Perfect anomaly map: every defect pixel scores higher than every normal pixel."""
    gt = np.zeros((16, 16), dtype=np.uint8)
    gt[4:8, 4:8] = 1
    amap = np.zeros((16, 16), dtype=np.float32)
    amap[4:8, 4:8] = 1.0  # defect scores are all 1.0, background is 0.0
    val = aupro([gt], [amap], fpr_limit=0.3)
    assert val is not None
    assert val == pytest.approx(1.0, abs=0.05)


def test_aupro_range():
    rng = np.random.default_rng(0)
    gt = (rng.random((20, 20)) > 0.8).astype(np.uint8)
    amap = rng.random((20, 20)).astype(np.float32)
    val = aupro([gt], [amap], fpr_limit=0.3)
    if val is not None:
        assert 0.0 <= val <= 1.0


# ---------------------------------------------------------------------------
# best_f1
# ---------------------------------------------------------------------------

def test_best_f1_perfect():
    labels = [0, 0, 1, 1]
    scores = [0.1, 0.2, 0.8, 0.9]
    f1, thr = best_f1(labels, scores)
    assert f1 == pytest.approx(1.0)


def test_best_f1_single_class_returns_zero():
    f1, thr = best_f1([1, 1, 1], [0.1, 0.5, 0.9])
    assert f1 == 0.0


def test_best_f1_range():
    labels = [0, 1, 0, 1, 0]
    scores = [0.3, 0.7, 0.2, 0.6, 0.4]
    f1, thr = best_f1(labels, scores)
    assert 0.0 <= f1 <= 1.0


# ---------------------------------------------------------------------------
# dice_iou
# ---------------------------------------------------------------------------

def test_dice_iou_empty_vs_empty():
    gt = np.zeros((10, 10), dtype=np.uint8)
    pred = np.zeros((10, 10), dtype=np.uint8)
    result = dice_iou(gt, pred)
    assert result["dice"] == pytest.approx(1.0)
    assert result["iou"] == pytest.approx(1.0)


def test_dice_iou_empty_gt_nonempty_pred():
    gt = np.zeros((10, 10), dtype=np.uint8)
    pred = np.ones((10, 10), dtype=np.uint8)
    result = dice_iou(gt, pred)
    assert result["dice"] == pytest.approx(0.0)
    assert result["iou"] == pytest.approx(0.0)


def test_dice_iou_nonempty_gt_empty_pred():
    gt = np.ones((10, 10), dtype=np.uint8)
    pred = np.zeros((10, 10), dtype=np.uint8)
    result = dice_iou(gt, pred)
    assert result["dice"] == pytest.approx(0.0)
    assert result["iou"] == pytest.approx(0.0)


def test_dice_iou_perfect():
    gt = np.zeros((10, 10), dtype=np.uint8)
    gt[3:7, 3:7] = 1
    result = dice_iou(gt, gt)
    assert result["dice"] == pytest.approx(1.0, abs=1e-5)
    assert result["iou"] == pytest.approx(1.0, abs=1e-5)


def test_dice_iou_partial_overlap():
    gt = np.zeros((10, 10), dtype=np.uint8)
    gt[0:5, 0:5] = 1
    pred = np.zeros((10, 10), dtype=np.uint8)
    pred[3:8, 3:8] = 1
    result = dice_iou(gt, pred)
    assert 0.0 < result["dice"] < 1.0
    assert 0.0 < result["iou"] < 1.0
    assert result["dice"] >= result["iou"]  # Dice >= IoU always


def test_dice_iou_returns_dict_keys():
    result = dice_iou(np.ones((4, 4)), np.ones((4, 4)))
    assert "dice" in result
    assert "iou" in result


# ---------------------------------------------------------------------------
# detection metrics (smoke test)
# ---------------------------------------------------------------------------

def test_coco_summary_structure():
    """coco_summary should return a dict with 12 float entries."""
    from kip.metrics.detection import coco_summary
    from unittest.mock import MagicMock
    import numpy as np

    mock_eval = MagicMock()
    mock_eval.stats = np.array([0.5] * 12)
    result = coco_summary(mock_eval)
    assert len(result) == 12
    assert all(isinstance(v, float) for v in result.values())
    assert "ap50_95" in result
    assert "ap50" in result
