"""Pytest fixtures reproducing real BGAD quirks for unit tests.

Quirks reproduced:
1. Duplicated masks across split dirs (train/masks ≡ val/masks).
2. Cross-split mask: mask in train/masks whose base image is only in val/base_images.
3. `_v2` stem: image tool10_..._0008_isolated_v2.jpg / mask ..._v2_polishing_wear.png.
4. Multi-mask image: one image with two masks (different defect types).
5. Empty YOLO label file (zero annotations).
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# BGAD fixture (synthetic mini-BGAD)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def bgad_root(tmp_path_factory):
    """Construct a minimal synthetic BGAD tree reproducing all documented quirks."""
    root = tmp_path_factory.mktemp("bgad")

    for split in ("train", "val"):
        (root / split / "base_images").mkdir(parents=True)
        (root / split / "masks").mkdir(parents=True)

    def _white_img(path: Path, hw: tuple[int, int] = (512, 512)):
        img = np.full((hw[0], hw[1], 3), 200, dtype=np.uint8)
        cv2.imwrite(str(path), img)

    def _mask(path: Path, hw: tuple[int, int] = (512, 512), filled: bool = True):
        mask = np.zeros(hw, dtype=np.uint8)
        if filled:
            mask[100:200, 100:200] = 255
        cv2.imwrite(str(path), mask)

    # ----- Train base images -----
    # good image (no mask)
    _white_img(root / "train" / "base_images" / "tool97_bevel_gear_spindle_closeup_0001_isolated.jpg")

    # defect image with ONE mask
    _white_img(root / "train" / "base_images" / "tool02_bevel_gear_spindle_closeup_0008_isolated.jpg")

    # defect image with TWO masks (quirk 4: multi-mask)
    _white_img(root / "train" / "base_images" / "tool09_bevel_gear_spindle_closeup_0013_isolated.jpg")

    # _v2 image (quirk 3)
    _white_img(root / "train" / "base_images" / "tool10_bevel_gear_spindle_closeup_0008_isolated_v2.jpg")

    # ----- Val base images -----
    # Image for the cross-split mask (quirk 2): base image is in VAL, mask will be in TRAIN
    _white_img(root / "val" / "base_images" / "tool08_bevel_gear_spindle_closeup_0014_isolated.jpg")

    # good val image
    _white_img(root / "val" / "base_images" / "tool99_bevel_gear_spindle_closeup_0003_isolated.jpg")

    # ----- Masks (identical in train/masks and val/masks — quirk 1: dedup) -----
    mask_files = {
        # matches train image tool02_0008 (single mask)
        "tool02_bevel_gear_spindle_closeup_0008_isolated_tooth_end_rupture.png": True,
        # mask 1 for multi-mask image (tool09_0013)
        "tool09_bevel_gear_spindle_closeup_0013_isolated_pitting.png": True,
        # mask 2 for multi-mask image (tool09_0013)
        "tool09_bevel_gear_spindle_closeup_0013_isolated_tooth_end_rupture.png": True,
        # _v2 mask (quirk 3)
        "tool10_bevel_gear_spindle_closeup_0008_isolated_v2_polishing_wear.png": True,
        # cross-split mask: image is in val, mask appears in train/masks (quirk 2)
        "tool08_bevel_gear_spindle_closeup_0014_isolated_fretting_corrosion.png": True,
    }

    for fname, filled in mask_files.items():
        for split in ("train", "val"):
            _mask(root / split / "masks" / fname, filled=filled)

    return root


# ---------------------------------------------------------------------------
# YOLO-seg fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def yolo_seg_dataset(tmp_path_factory):
    """Minimal YOLO-seg dataset with edge cases.

    Contains:
    - One normal label file with two instances
    - One image with NO label file (zero annotations)
    - One image with an EMPTY label file (quirk 5)
    - One image with a <3-point polygon line (should be skipped)
    """
    root = tmp_path_factory.mktemp("yolo_seg")
    imgs = root / "images"
    lbls = root / "labels"
    imgs.mkdir()
    lbls.mkdir()

    W, H = 640, 480

    def _img(name):
        img = np.full((H, W, 3), 128, dtype=np.uint8)
        cv2.imwrite(str(imgs / name), img)

    def _lbl(name, content: str):
        (lbls / name).write_text(content)

    # Image 1: two valid annotations
    _img("img_001.jpg")
    _lbl(
        "img_001.txt",
        "0 0.1 0.1 0.9 0.1 0.9 0.9 0.1 0.9\n"
        "3 0.2 0.2 0.8 0.2 0.8 0.8 0.2 0.8\n",
    )

    # Image 2: no label file -> zero annotations
    _img("img_002.jpg")

    # Image 3: empty label file (quirk 5)
    _img("img_003.jpg")
    _lbl("img_003.txt", "")

    # Image 4: degenerate line (<3 points) -> should be skipped
    _img("img_004.jpg")
    _lbl("img_004.txt", "0 0.5 0.5 0.6 0.6\n")  # only 2 points

    return root, W, H


# ---------------------------------------------------------------------------
# Manifest dataframe fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_manifest(bgad_root):
    """Build a manifest from the synthetic bgad_root fixture."""
    from kip.data.bgad_manifest import build_manifest
    return build_manifest(bgad_root)
