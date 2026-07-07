"""Contract tests: every Stage-2 method fits on good tiles and returns a
correctly-shaped, deterministic (amap, score)."""
import numpy as np
import pytest

from kip.config import Stage2Config
from kip.stage2.autoencoder import ConvAE
from kip.stage2.padim import PaDiM
from kip.stage2.patchcore import PatchCore


def _tiles(n=4, size=576, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.integers(0, 255, (size, size, 3), dtype=np.uint8) for _ in range(n)]


@pytest.mark.parametrize("cls", [PatchCore, PaDiM, ConvAE])
def test_fit_score_shape(cls):
    cfg = Stage2Config(epochs=1, seed=0, device="cpu")
    m = cls(cfg, "cpu")
    m.fit(_tiles())
    tile = _tiles(1)[0]
    amap, score = m.score(tile)
    assert amap.shape == tile.shape[:2]
    assert amap.dtype == np.float32
    assert isinstance(score, float)


@pytest.mark.parametrize("cls", [PatchCore, PaDiM])
def test_deterministic(cls):
    cfg = Stage2Config(seed=0, device="cpu")
    tile = _tiles(1, seed=99)[0]
    outs = []
    for _ in range(2):
        m = cls(cfg, "cpu")
        m.fit(_tiles(seed=0))
        outs.append(m.score(tile)[1])
    assert outs[0] == pytest.approx(outs[1], rel=1e-4)


def test_unet_supervised_contract():
    from kip.stage2.unet import UNetSupervised
    cfg = Stage2Config(method="unet", epochs=1, seed=0, device="cpu")
    m = UNetSupervised(cfg, "cpu")
    tiles = _tiles(4, size=256)
    masks = [np.zeros((256, 256), np.uint8) for _ in range(4)]
    masks[0][50:100, 50:100] = 1
    m.fit_supervised(tiles, masks)
    amap, score = m.score(tiles[0])
    assert amap.shape == (256, 256)
    assert 0.0 <= score <= 1.0
