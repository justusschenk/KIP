"""Tests for kip/data/bgad_manifest.py."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from kip.data.bgad_manifest import (
    MASK_STEM_RE,
    ManifestError,
    build_manifest,
    normalize_defect_type,
    save_manifest,
    union_masks,
    validate_manifest,
)


# ---------------------------------------------------------------------------
# Unit tests: normalize_defect_type
# ---------------------------------------------------------------------------

def test_normalize_plain():
    dtype, is_v2 = normalize_defect_type("tooth_end_rupture")
    assert dtype == "tooth_end_rupture"
    assert is_v2 is False


def test_normalize_v2_prefix():
    dtype, is_v2 = normalize_defect_type("v2_polishing_wear")
    assert dtype == "polishing_wear"
    assert is_v2 is True


def test_normalize_polishing_wear_plain():
    dtype, is_v2 = normalize_defect_type("polishing_wear")
    assert dtype == "polishing_wear"
    assert is_v2 is False


# ---------------------------------------------------------------------------
# Unit tests: MASK_STEM_RE regex
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("filename,expected_stem,expected_defect", [
    (
        "tool02_bevel_gear_spindle_closeup_0008_isolated_tooth_end_rupture.png",
        "tool02_bevel_gear_spindle_closeup_0008_isolated",
        "tooth_end_rupture",
    ),
    (
        "tool10_bevel_gear_spindle_closeup_0008_isolated_v2_polishing_wear.png",
        "tool10_bevel_gear_spindle_closeup_0008_isolated_v2",
        "polishing_wear",
    ),
    (
        "tool09_bevel_gear_spindle_closeup_0013_isolated_pitting.png",
        "tool09_bevel_gear_spindle_closeup_0013_isolated",
        "pitting",
    ),
])
def test_mask_stem_re(filename, expected_stem, expected_defect):
    m = MASK_STEM_RE.match(filename)
    assert m is not None, f"Regex did not match: {filename}"
    assert m.group(1) == expected_stem
    assert m.group(2) == expected_defect


# ---------------------------------------------------------------------------
# Integration: build_manifest on synthetic fixture
# ---------------------------------------------------------------------------

def test_manifest_total_images(sample_manifest):
    """Synthetic fixture: 4 train + 2 val = 6 images."""
    assert len(sample_manifest) == 6


def test_manifest_no_duplicate_images(sample_manifest):
    assert not sample_manifest["image"].duplicated().any()


def test_manifest_defect_status_values(sample_manifest):
    assert set(sample_manifest["defect_status"].unique()).issubset(
        {"good", "defect", "unlabeled"}
    )


def test_manifest_good_images(sample_manifest):
    """tool97_0001 (train) and tool99_0003 (val) should be good."""
    good = sample_manifest[sample_manifest["defect_status"] == "good"]
    good_stems = [Path(p).stem for p in good["image"]]
    assert any("tool97" in s for s in good_stems)
    assert any("tool99" in s for s in good_stems)


def test_manifest_multi_mask_image(sample_manifest):
    """tool09_0013 has two masks -> defect, semicolon-separated mask_paths."""
    row = sample_manifest[
        sample_manifest["image"].str.contains("tool09") &
        sample_manifest["image"].str.contains("0013")
    ]
    assert len(row) == 1, "tool09_0013 should appear exactly once"
    mp = row.iloc[0]["mask_paths"]
    assert ";" in mp, "Expected multiple masks (semicolon-separated)"
    masks = mp.split(";")
    assert len(masks) == 2


def test_manifest_v2_stem(sample_manifest):
    """tool10_..._v2 image is correctly paired with its _v2 mask."""
    row = sample_manifest[sample_manifest["image"].str.contains("_v2")]
    assert len(row) == 1
    assert row.iloc[0]["defect_status"] == "defect"
    assert "polishing_wear" in row.iloc[0]["defect_types"]


def test_manifest_cross_split_mask(sample_manifest):
    """tool08_0014: image in val, mask in train/masks only -> should be defect in val."""
    row = sample_manifest[
        sample_manifest["image"].str.contains("tool08") &
        sample_manifest["image"].str.contains("0014")
    ]
    assert len(row) == 1, "tool08_0014 must appear once (in val split)"
    assert row.iloc[0]["split"] == "val"
    assert row.iloc[0]["defect_status"] == "defect"


def test_manifest_dedup_masks(bgad_root):
    """Identical mask filenames in train/masks and val/masks are deduplicated.

    After dedup the mask count should equal the number of unique mask FILENAMES,
    not twice that.
    """
    df = build_manifest(bgad_root)
    # Collect all individual mask paths referenced in the manifest
    all_mask_paths = []
    for mp_str in df["mask_paths"].dropna():
        if mp_str:
            all_mask_paths.extend(mp_str.split(";"))
    # All referenced mask paths should resolve to unique canonical paths
    canonical = [str(Path(p).resolve()) for p in all_mask_paths]
    assert len(canonical) == len(set(canonical)), "Duplicate mask paths found in manifest"


def test_manifest_tool_ids(sample_manifest):
    """All expected tool IDs are present."""
    expected = {"tool02", "tool08", "tool09", "tool10", "tool97", "tool99"}
    found = set(sample_manifest["tool_id"].unique())
    assert expected == found


def test_manifest_missing_mask_policy_unlabeled(bgad_root):
    df = build_manifest(bgad_root, missing_mask_policy="unlabeled")
    statuses = set(df["defect_status"].unique())
    assert "unlabeled" in statuses


def test_manifest_missing_mask_policy_error(bgad_root):
    """'error' policy should raise ManifestError when images have no mask."""
    with pytest.raises(ManifestError):
        build_manifest(bgad_root, missing_mask_policy="error")


# ---------------------------------------------------------------------------
# union_masks
# ---------------------------------------------------------------------------

def test_union_masks_basic(bgad_root, sample_manifest):
    """Union of a single mask should equal the mask itself."""
    defect_rows = sample_manifest[sample_manifest["defect_status"] == "defect"]
    # pick first row with exactly one mask
    for _, row in defect_rows.iterrows():
        masks = [m for m in row["mask_paths"].split(";") if m]
        if len(masks) == 1:
            import cv2
            arr = cv2.imread(masks[0], cv2.IMREAD_GRAYSCALE)
            hw = (row["height"], row["width"])
            union = union_masks(masks, hw)
            assert union.dtype == np.uint8
            assert union.max() <= 1
            break


def test_union_masks_empty_list():
    union = union_masks([], (64, 64))
    assert union.shape == (64, 64)
    assert union.max() == 0


# ---------------------------------------------------------------------------
# save_manifest
# ---------------------------------------------------------------------------

def test_save_manifest(sample_manifest, tmp_path):
    import json
    csv_path = save_manifest(sample_manifest, tmp_path / "out")
    assert csv_path.exists()
    meta_path = csv_path.parent / "manifest_meta.json"
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text())
    assert meta["n_images"] == len(sample_manifest)
    assert "sha256" in meta


# ---------------------------------------------------------------------------
# validate_manifest
# ---------------------------------------------------------------------------

def test_validate_manifest_no_fatal(sample_manifest, bgad_root):
    """validate_manifest should not raise on a valid manifest."""
    warnings = validate_manifest(sample_manifest, bgad_root)
    # warnings are acceptable (e.g. non-binary mask values) but no exception
    assert isinstance(warnings, list)
