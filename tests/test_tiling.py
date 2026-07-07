"""Tests for kip/data/tiling.py."""
from __future__ import annotations

import numpy as np
import pytest

from kip.config import TilingConfig
from kip.data.tiling import Tile, compute_grid, is_background_tile, stitch, tile_image


# ---------------------------------------------------------------------------
# compute_grid
# ---------------------------------------------------------------------------

def test_compute_grid_covers_full_image():
    """All pixel columns and rows should be covered by at least one tile."""
    H, W, TILE = 1024, 1024, 576
    corners = compute_grid(H, W, tile=TILE, overlap=0.25)
    assert len(corners) > 0
    covered_rows = set()
    covered_cols = set()
    for y0, x0 in corners:
        covered_rows.update(range(y0, min(H, y0 + TILE)))
        covered_cols.update(range(x0, min(W, x0 + TILE)))
    assert covered_rows == set(range(H))
    assert covered_cols == set(range(W))


def test_compute_grid_no_duplicates():
    corners = compute_grid(512, 512, tile=256, overlap=0.25)
    assert len(corners) == len(set(corners))


def test_compute_grid_small_image():
    """Image smaller than tile -> single tile at (0, 0)."""
    corners = compute_grid(100, 100, tile=576, overlap=0.25)
    assert (0, 0) in corners


# ---------------------------------------------------------------------------
# is_background_tile
# ---------------------------------------------------------------------------

def test_is_background_tile_white():
    white = np.full((64, 64, 3), 250, dtype=np.uint8)
    assert is_background_tile(white, white_thresh=240, white_frac=0.95)


def test_is_background_tile_dark():
    dark = np.zeros((64, 64, 3), dtype=np.uint8)
    assert not is_background_tile(dark, white_thresh=240, white_frac=0.95)


def test_is_background_tile_mixed():
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    # Make 96% of pixels near-white -> background
    img[:62, :] = 250
    assert is_background_tile(img, white_thresh=240, white_frac=0.95)


# ---------------------------------------------------------------------------
# tile_image
# ---------------------------------------------------------------------------

def _make_cfg(**kwargs) -> TilingConfig:
    defaults = dict(tile_size=256, overlap=0.25, white_thresh=240, white_frac=0.95, min_fg_coverage=0.99)
    defaults.update(kwargs)
    return TilingConfig(**defaults)


def test_tile_image_returns_list():
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    cfg = _make_cfg()
    tiles = tile_image(img, None, cfg)
    assert isinstance(tiles, list)
    # Dark image: no tiles should be dropped as background
    assert len(tiles) > 0


def test_tile_image_drops_white_background():
    """A fully white image should produce zero kept tiles."""
    img = np.full((512, 512, 3), 250, dtype=np.uint8)
    cfg = _make_cfg()
    tiles = tile_image(img, None, cfg)
    assert len(tiles) == 0


def test_tile_image_mask_consistent():
    """Tile mask shape should match tile image shape."""
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    mask = np.zeros((512, 512), dtype=np.uint8)
    mask[100:200, 100:200] = 1
    cfg = _make_cfg()
    tiles = tile_image(img, mask, cfg)
    for tile_meta, tile_img, tile_mask in tiles:
        assert tile_img.shape[:2] == tile_mask.shape


def test_tile_image_coverage_gt_positive(tmp_path):
    """Tiles must cover >= 99% of GT positive pixels (Risk-5)."""
    H, W = 576, 576
    img = np.zeros((H, W, 3), dtype=np.uint8)
    # Small defect region in corner — img is all dark so no bg filtering
    mask = np.zeros((H, W), dtype=np.uint8)
    mask[10:50, 10:50] = 1
    cfg = _make_cfg(tile_size=256, overlap=0.25, min_fg_coverage=0.99)
    tiles = tile_image(img, mask, cfg)
    # Verify coverage
    covered = np.zeros((H, W), dtype=np.uint8)
    for tile_meta, _, tm in tiles:
        y0, x0 = tile_meta.y0, tile_meta.x0
        covered[y0:y0 + tile_meta.h, x0:x0 + tile_meta.w] = np.maximum(
            covered[y0:y0 + tile_meta.h, x0:x0 + tile_meta.w],
            mask[y0:y0 + tile_meta.h, x0:x0 + tile_meta.w],
        )
    total_fg = mask.sum()
    covered_fg = covered.sum()
    assert covered_fg / total_fg >= 0.99


# ---------------------------------------------------------------------------
# stitch
# ---------------------------------------------------------------------------

def test_stitch_mean_no_overlap():
    """Non-overlapping tiles should produce the original values."""
    H, W = 256, 256
    tile1 = Tile("", 0, 0, 128, 128, 0.0)
    tile2 = Tile("", 128, 0, 128, 128, 0.0)
    tile3 = Tile("", 0, 128, 128, 128, 0.0)
    tile4 = Tile("", 128, 128, 128, 128, 0.0)

    s1 = np.full((128, 128), 0.1, dtype=np.float32)
    s2 = np.full((128, 128), 0.2, dtype=np.float32)
    s3 = np.full((128, 128), 0.3, dtype=np.float32)
    s4 = np.full((128, 128), 0.4, dtype=np.float32)

    tiles = [(tile1, s1), (tile2, s2), (tile3, s3), (tile4, s4)]
    result = stitch(tiles, (H, W), mode="mean")

    assert result.shape == (H, W)
    assert abs(result[0, 0] - 0.1) < 1e-5
    assert abs(result[0, 130] - 0.2) < 1e-5


def test_stitch_max_mode():
    H, W = 64, 64
    tile1 = Tile("", 0, 0, 64, 64, 0.0)
    tile2 = Tile("", 0, 0, 64, 64, 0.0)
    s1 = np.full((64, 64), 0.3, dtype=np.float32)
    s2 = np.full((64, 64), 0.7, dtype=np.float32)
    result = stitch([(tile1, s1), (tile2, s2)], (H, W), mode="max")
    assert np.allclose(result, 0.7)


def test_stitch_empty_tiles():
    result = stitch([], (64, 64), fill_value=0.0)
    assert result.shape == (64, 64)
    assert result.max() == 0.0
