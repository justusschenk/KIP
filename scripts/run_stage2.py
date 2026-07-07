"""CLI: run a Stage-2 defect-detection method and write a benchmark run dir.

Examples:
    python scripts/run_stage2.py --method patchcore --split loto --smoke --device mps
    python scripts/run_stage2.py --method unet --aug on --split loto --epochs 200 --device cuda:0
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kip.config import Stage2Config          # noqa: E402
from kip.hardware import pick_device         # noqa: E402
from kip.stage2.runner import run_stage2     # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["patchcore", "padim", "ae", "unet"], required=True)
    ap.add_argument("--split", choices=["loto", "gkf", "fixed"], default="loto")
    ap.add_argument("--aug", choices=["on", "off"], default="off")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--tile", type=int, default=None)
    ap.add_argument("--loss", choices=["bce_dice", "focal"], default="bce_dice")
    ap.add_argument("--imgsz", type=int, default=1024, help="Stage-1 crop inference size")
    args = ap.parse_args()

    device = pick_device(args.device)
    tile = args.tile if args.tile is not None else (128 if args.smoke else 256)
    epochs = args.epochs if args.epochs is not None else (3 if args.smoke else 50)
    imgsz = 640 if args.smoke else args.imgsz

    cfg = Stage2Config(
        method=args.method, split=args.split, augmentation=(args.aug == "on"),
        epochs=epochs, tile_size=tile, loss=args.loss,
        device=device, seed=args.seed, smoke=args.smoke,
    )
    run_dir = run_stage2(cfg, imgsz=imgsz)
    import json
    m = json.loads((run_dir / "metrics.json").read_text())["metrics"]["pooled"]
    print(f"\n[{args.method}/{args.split}] run -> {run_dir.name}")
    print(f"  image_auroc={m['image_auroc']}  pixel_auroc={m['pixel_auroc']}  "
          f"aupro={m['pixel_aupro']}  dice={m['dice']}")


if __name__ == "__main__":
    main()
