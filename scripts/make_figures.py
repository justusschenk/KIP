"""Generate comparison figures from the benchmark summary.csv files.

Reads results/{component_benchmark,defect_detection}/summary.csv and writes
paper figures to results/figures/. Smoke-sourced rows are labelled.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
FIG = REPO / "results" / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def _grouped_bar(df, value_col, group_col, cat_col, title, ylabel, out):
    cats = list(dict.fromkeys(df[cat_col]))
    groups = list(dict.fromkeys(df[group_col]))
    import numpy as np
    x = np.arange(len(cats))
    w = 0.8 / max(1, len(groups))
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, g in enumerate(groups):
        sub = df[df[group_col] == g]
        vals = [sub[sub[cat_col] == c][value_col].mean() if not sub[sub[cat_col] == c].empty
                else 0 for c in cats]
        ax.bar(x + i * w, vals, w, label=str(g))
    ax.set_xticks(x + w * (len(groups) - 1) / 2)
    ax.set_xticklabels(cats, rotation=0)
    ax.set_ylabel(ylabel); ax.set_title(title); ax.set_ylim(0, 1.05)
    ax.legend(title=group_col); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print("wrote", out)


def stage2():
    p = REPO / "results/defect_detection/summary.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    smoke = "  (SMOKE)" if df["smoke"].all() else ""
    for metric, name in [("metric.image_auroc", "Bild-AUROC"),
                         ("metric.pixel_auroc", "Pixel-AUROC")]:
        _grouped_bar(df, metric, "split_scheme", "model",
                     f"Stage 2 Defekterkennung — {name}{smoke}", name,
                     FIG / f"stage2_{metric.split('.')[-1]}.png")


def stage1():
    p = REPO / "results/component_benchmark/summary.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    df["cfg"] = df["model"] + "_aug" + df["augmentation"].astype(str)
    smoke = "  (SMOKE)" if df["smoke"].all() else ""
    _grouped_bar(df, "metric.segm_map50", "model", "cfg",
                 f"Stage 1 Komponentensegmentierung — Mask mAP@50{smoke}",
                 "segm mAP@50", FIG / "stage1_segm_map50.png")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "stage1"):
        stage1()
    if which in ("all", "stage2"):
        stage2()
