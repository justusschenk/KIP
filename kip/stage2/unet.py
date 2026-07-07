"""Supervised binary defect segmentation (U-Net-ResNet18, few-shot).

Methodology: supervised segmentation using the delivered defect masks. This is
the complementary supervised counterpart to the three unsupervised methods.
Only 2-5 examples per defect type exist, so this is trained BINARY (defect vs
not) — no multi-class claims (Risk-2, brief).
"""
from __future__ import annotations

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F


def _dice_loss(logits, target, eps=1.0):
    prob = torch.sigmoid(logits)
    num = 2 * (prob * target).sum(dim=(1, 2, 3)) + eps
    den = (prob + target).sum(dim=(1, 2, 3)) + eps
    return (1 - num / den).mean()


class UNetSupervised:
    """Supervised segmentation; shares the score() signature of AnomalyMethod."""

    name = "unet"

    def __init__(self, cfg, device: str = "cpu"):
        self.cfg = cfg
        self.device = device
        self.size = 256
        self.loss_name = getattr(cfg, "loss", "bce_dice")
        self.model = smp.Unet(
            encoder_name="resnet18", encoder_weights="imagenet",
            in_channels=3, classes=1,
        ).to(device)

    def _x(self, tile):
        rgb = cv2.cvtColor(cv2.resize(tile, (self.size, self.size)), cv2.COLOR_BGR2RGB)
        return torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0

    def _y(self, mask):
        m = cv2.resize(mask.astype(np.uint8), (self.size, self.size),
                       interpolation=cv2.INTER_NEAREST)
        return torch.from_numpy((m > 0).astype(np.float32))[None]

    def _loss(self, logits, target, pos_weight):
        if self.loss_name == "focal":
            return smp.losses.FocalLoss(mode="binary")(logits, target)
        bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
        return bce + _dice_loss(logits, target)

    def fit_supervised(self, train_tiles, train_masks, val=None) -> None:
        X = torch.stack([self._x(t) for t in train_tiles])
        Y = torch.stack([self._y(m) for m in train_masks])
        # oversample positive tiles to counter extreme imbalance (Risk-6)
        pos = np.array([float(y.sum() > 0) for y in Y])
        weights = np.where(pos > 0, 3.0, 1.0)
        prob = weights / weights.sum()
        pix = Y.mean().clamp(min=1e-4)
        pos_weight = ((1 - pix) / pix).to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=getattr(self.cfg, "lr", 1e-3))
        epochs = getattr(self.cfg, "epochs", 50)
        bs = max(1, min(getattr(self.cfg, "batch", 8), X.shape[0]))
        rng = np.random.default_rng(getattr(self.cfg, "seed", 42))
        n = X.shape[0]
        self.model.train()
        for _ in range(epochs):
            for _ in range(max(1, n // bs)):
                idx = rng.choice(n, size=bs, p=prob)
                xb = X[idx].to(self.device)
                yb = Y[idx].to(self.device)
                opt.zero_grad()
                loss = self._loss(self.model(xb), yb, pos_weight)
                loss.backward()
                opt.step()
        self.model.eval()

    @torch.no_grad()
    def score(self, tile) -> tuple[np.ndarray, float]:
        x = self._x(tile).unsqueeze(0).to(self.device)
        prob = torch.sigmoid(self.model(x))[0, 0].cpu().numpy()
        amap = cv2.resize(prob, (tile.shape[1], tile.shape[0]))
        return amap.astype(np.float32), float(amap.max())
