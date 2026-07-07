"""Tests for kip/data/splits.py."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kip.data.splits import (
    Fold,
    assert_no_tool_leakage,
    fixed_split,
    group_kfold,
    leave_one_tool_out,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_manifest(tool_split_defect: list[tuple[str, str, str]]) -> pd.DataFrame:
    """Build a minimal manifest DataFrame."""
    rows = []
    for i, (tool, split, status) in enumerate(tool_split_defect):
        rows.append({
            "image": f"/fake/img_{i}.jpg",
            "tool_id": tool,
            "split": split,
            "defect_status": status,
        })
    return pd.DataFrame(rows)


SAMPLE_MANIFEST = _make_manifest([
    ("tool02", "train", "defect"),
    ("tool02", "train", "good"),
    ("tool02", "val",   "defect"),
    ("tool03", "val",   "defect"),
    ("tool08", "train", "defect"),
    ("tool08", "val",   "defect"),
    ("tool09", "train", "defect"),
    ("tool09", "val",   "defect"),
    ("tool10", "train", "defect"),
    ("tool10", "val",   "defect"),
    ("tool97", "train", "good"),
    ("tool97", "val",   "good"),
    ("tool99", "train", "good"),
    ("tool99", "val",   "good"),
])


# ---------------------------------------------------------------------------
# leave_one_tool_out
# ---------------------------------------------------------------------------

def test_loto_n_folds():
    folds = leave_one_tool_out(SAMPLE_MANIFEST)
    n_tools = SAMPLE_MANIFEST["tool_id"].nunique()
    assert len(folds) == n_tools


def test_loto_each_fold_has_name():
    folds = leave_one_tool_out(SAMPLE_MANIFEST)
    for fold in folds:
        assert fold.name.startswith("loto_")


def test_loto_held_out_tools_correct():
    folds = leave_one_tool_out(SAMPLE_MANIFEST)
    all_tools = set(SAMPLE_MANIFEST["tool_id"].unique())
    for fold in folds:
        assert len(fold.held_out_tools) == 1
        assert fold.held_out_tools[0] in all_tools


def test_loto_no_leakage():
    folds = leave_one_tool_out(SAMPLE_MANIFEST)
    for fold in folds:
        assert_no_tool_leakage(fold, SAMPLE_MANIFEST)


def test_loto_all_indices_covered():
    folds = leave_one_tool_out(SAMPLE_MANIFEST)
    all_test_idx = set()
    for fold in folds:
        all_test_idx.update(fold.test_idx.tolist())
    assert all_test_idx == set(range(len(SAMPLE_MANIFEST)))


def test_loto_degenerate_fold():
    """tool03 has only one defect image -> fold is valid but image_auroc should return None."""
    folds = leave_one_tool_out(SAMPLE_MANIFEST)
    tool03_fold = next(f for f in folds if "tool03" in f.held_out_tools)
    test_df = SAMPLE_MANIFEST.iloc[tool03_fold.test_idx]
    assert len(test_df) == 1
    assert len(test_df["defect_status"].unique()) == 1  # single class


# ---------------------------------------------------------------------------
# group_kfold
# ---------------------------------------------------------------------------

def test_gkf_n_folds():
    folds = group_kfold(SAMPLE_MANIFEST, n_splits=3, seed=42)
    assert len(folds) == 3


def test_gkf_no_leakage():
    folds = group_kfold(SAMPLE_MANIFEST, n_splits=3, seed=42)
    for fold in folds:
        assert_no_tool_leakage(fold, SAMPLE_MANIFEST)


def test_gkf_names():
    folds = group_kfold(SAMPLE_MANIFEST, n_splits=3, seed=42)
    for fold in folds:
        assert fold.name.startswith("gkf_")


# ---------------------------------------------------------------------------
# fixed_split
# ---------------------------------------------------------------------------

def test_fixed_split_returns_fold():
    fold = fixed_split(SAMPLE_MANIFEST)
    assert isinstance(fold, Fold)
    assert fold.name == "fixed"


def test_fixed_split_train_val():
    fold = fixed_split(SAMPLE_MANIFEST)
    train_splits = SAMPLE_MANIFEST.iloc[fold.train_idx]["split"].unique()
    test_splits = SAMPLE_MANIFEST.iloc[fold.test_idx]["split"].unique()
    assert set(train_splits) == {"train"}
    assert set(test_splits) == {"val"}


def test_fixed_split_same_tool_in_both_splits():
    """BGAD fixed split has the SAME tools in train and val by design.

    assert_no_tool_leakage() should RAISE for fixed_split because tools like
    tool02, tool08, tool09 etc. appear in both splits. The fixed split is
    the only strategy where tool leakage is expected and acceptable.
    """
    fold = fixed_split(SAMPLE_MANIFEST)
    # Verify that the leakage IS detected (expected for BGAD fixed split)
    with pytest.raises(ValueError, match="leakage"):
        assert_no_tool_leakage(fold, SAMPLE_MANIFEST)


# ---------------------------------------------------------------------------
# assert_no_tool_leakage
# ---------------------------------------------------------------------------

def test_leakage_detected():
    """Artificially create a leaking fold and verify detection."""
    # tool02 appears in both train and val -> leakage
    bad_fold = Fold(
        name="bad",
        train_idx=np.array([0, 1, 2]),   # tool02 train + val
        test_idx=np.array([2]),           # tool02 val (also in train)
        held_out_tools=["tool02"],
    )
    with pytest.raises(ValueError, match="leakage"):
        assert_no_tool_leakage(bad_fold, SAMPLE_MANIFEST)


def test_no_leakage_passes():
    # Only tool03 in test
    fold = Fold(
        name="ok",
        train_idx=np.array([0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]),
        test_idx=np.array([3]),
        held_out_tools=["tool03"],
    )
    assert_no_tool_leakage(fold, SAMPLE_MANIFEST)  # should not raise
