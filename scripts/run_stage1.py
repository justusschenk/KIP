#!/usr/bin/env python
"""Stage-1 component-segmentation benchmark runner.

Trains YOLO11n-seg or Mask2Former (Swin-T) on the real_v3 dataset,
evaluates on the fixed test split via the single shared COCO evaluator,
and writes schema-conformant artefacts to results/component_benchmark/.

Usage examples
--------------
# Smoke run – YOLO with aug on, MPS
.venv/bin/python scripts/run_stage1.py --model yolo --aug on --smoke --device mps

# Smoke run – Mask2Former aug off, CPU
.venv/bin/python scripts/run_stage1.py --model mask2former --aug off --smoke --device cpu

# Full run (CUDA server)
python scripts/run_stage1.py --model yolo --aug on --epochs 100 --imgsz 1088 \\
    --batch 16 --device cuda:0 --seed 42
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root on sys.path so that `kip` is importable when run from anywhere
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from kip import CLASS_NAMES
from kip.config import Stage1Config, seed_everything
from kip.hardware import hardware_info
from kip.reporting.results_io import (
    append_summary,
    create_run_dir,
    save_run,
)
from kip.stage1.evaluator import evaluate_coco

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_DATA_ROOT = _ROOT / "data" / "object_segmentation_real_v3_1088"
_DATA_YAML = _DATA_ROOT / "data.yaml"
_COCO_ROOT = _ROOT / "data" / "coco_converted"
_TRAIN_JSON = _COCO_ROOT / "train.json"
_VAL_JSON = _COCO_ROOT / "val.json"
_TEST_JSON = _COCO_ROOT / "test.json"
_IMAGES_TRAIN = _DATA_ROOT / "images" / "train"
_IMAGES_VAL = _DATA_ROOT / "images" / "val"
_IMAGES_TEST = _DATA_ROOT / "images" / "test"
_RESULTS_BASE = _ROOT / "results" / "component_benchmark"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage-1 component-segmentation benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", choices=["yolo", "mask2former"], required=True)
    p.add_argument("--aug", choices=["on", "off"], default="on",
                   help="Data augmentation toggle")
    p.add_argument("--smoke", action="store_true",
                   help="Smoke run: fewer epochs, fewer images, smoke=true in metrics")
    p.add_argument("--device", default="cpu",
                   help="Torch device string, e.g. cpu / mps / cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=None,
                   help="Override epoch count (default: 2 for smoke, 100 for full)")
    p.add_argument("--imgsz", type=int, default=None,
                   help="Inference/training image size (default: 320 smoke, 1088 full)")
    p.add_argument("--batch", type=int, default=None,
                   help="Batch size (default: 4 smoke, 16 full for YOLO; 2 smoke, 8 full for M2F)")
    p.add_argument("--lr", type=float, default=1e-4,
                   help="Learning rate (Mask2Former only)")
    p.add_argument("--freeze-backbone-epochs", type=int, default=0,
                   help="Epochs to keep Swin backbone frozen (Mask2Former only)")
    p.add_argument("--weights", type=str, default="yolo11n-seg.pt",
                   help="YOLO initial weights path or name")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> None:
    args = _parse_args(argv)

    # ------------------------------------------------------------------
    # Defaults per smoke / full / model
    # ------------------------------------------------------------------
    aug_on = args.aug == "on"

    if args.smoke:
        epochs = args.epochs or 2
        imgsz = args.imgsz or 320
        batch = args.batch or (4 if args.model == "yolo" else 2)
    else:
        epochs = args.epochs or 100
        imgsz = args.imgsz or (1088 if args.model == "yolo" else 800)
        batch = args.batch or (16 if args.model == "yolo" else 8)

    cfg = Stage1Config(
        model=args.model,
        augmentation=aug_on,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        lr=args.lr,
        freeze_backbone_epochs=args.freeze_backbone_epochs,
        device=args.device,
        seed=args.seed,
        smoke=args.smoke,
    )

    seed_everything(cfg.seed)

    # ------------------------------------------------------------------
    # Verify data
    # ------------------------------------------------------------------
    for p in [_DATA_YAML, _TRAIN_JSON, _VAL_JSON, _TEST_JSON]:
        if not p.exists():
            print(f"ERROR: required file not found: {p}", file=sys.stderr)
            sys.exit(1)

    # ------------------------------------------------------------------
    # Create output run directory
    # ------------------------------------------------------------------
    run_name = f"{args.model}_aug{args.aug}"
    run_dir = create_run_dir(_RESULTS_BASE, run_name)
    run_id = run_dir.name
    print(f"[stage1] run_id: {run_id}")
    print(f"[stage1] run_dir: {run_dir}")

    # ------------------------------------------------------------------
    # Dataset counts (from COCO JSONs)
    # ------------------------------------------------------------------
    def _count(json_path: Path) -> int:
        return len(json.loads(json_path.read_text())["images"])

    n_train = _count(_TRAIN_JSON)
    n_val = _count(_VAL_JSON)
    n_test = _count(_TEST_JSON)

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    t_train_start = time.time()

    pred_json = run_dir / "predictions" / "coco_predictions.json"
    ckpt_path = None

    if args.model == "yolo":
        from kip.stage1.yolo_trainer import YoloSegTrainer

        trainer = YoloSegTrainer(
            cfg=cfg,
            data_yaml=_DATA_YAML,
            run_dir=run_dir,
            weights=args.weights,
        )
        ckpt_path = trainer.train()
        train_seconds = time.time() - t_train_start
        print(f"[stage1] YOLO training done in {train_seconds:.1f}s")

        # Predict on test split
        t_infer = time.time()
        trainer.predict_to_coco(
            ckpt=ckpt_path,
            coco_gt_json=_TEST_JSON,
            images_dir=_IMAGES_TEST,
            out_json=pred_json,
        )
        infer_ms = (time.time() - t_infer) / n_test * 1000

    elif args.model == "mask2former":
        from kip.stage1.mask2former_trainer import Mask2FormerTrainer

        # Mask2Former: both train and val images live in separate dirs
        # The trainer images_dir is used for training; predict_to_coco gets
        # its own images_dir argument.
        trainer = Mask2FormerTrainer(
            cfg=cfg,
            coco_train_json=_TRAIN_JSON,
            coco_val_json=_VAL_JSON,
            images_dir=_IMAGES_TRAIN,
            run_dir=run_dir,
        )
        ckpt_path = trainer.train()
        train_seconds = time.time() - t_train_start
        print(f"[stage1] Mask2Former training done in {train_seconds:.1f}s")

        # Predict on test split
        t_infer = time.time()
        trainer.predict_to_coco(
            ckpt_dir=ckpt_path,
            coco_gt_json=_TEST_JSON,
            images_dir=_IMAGES_TEST,
            out_json=pred_json,
        )
        infer_ms = (time.time() - t_infer) / n_test * 1000

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------
    print(f"[stage1] Evaluating predictions at {pred_json}")
    metrics = evaluate_coco(
        gt_json=_TEST_JSON,
        pred_json=pred_json,
        class_names=CLASS_NAMES,
    )
    print(
        f"[stage1] segm_mAP50={metrics['segm_map50']:.4f}  "
        f"segm_mAP50_95={metrics['segm_map50_95']:.4f}  "
        f"bbox_mAP50={metrics['bbox_map50']:.4f}"
    )

    # ------------------------------------------------------------------
    # Build metrics.json envelope (§3 schema)
    # ------------------------------------------------------------------
    metrics_payload = {
        "schema_version": "1.0",
        "run_id": run_id,
        "stage": 1,
        "model": args.model,
        "augmentation": aug_on,
        "seed": cfg.seed,
        "smoke": cfg.smoke,
        "device": cfg.device,
        "split": {
            "scheme": "fixed",
            "n_folds": 1,
            "test_set": "real_v3/test",
        },
        "dataset": {
            "n_train": n_train,
            "n_val": n_val,
            "n_test": n_test,
            "n_good": 0,
            "n_defect": 0,
            "n_tiles_train": 0,
            "n_tiles_test": 0,
            "missing_mask_policy": "normal",
            "crop_failures": 0,
        },
        "runtime": {
            "train_seconds": round(train_seconds, 2),
            "infer_ms_per_image": round(infer_ms, 2),
        },
        "metrics": metrics,
    }

    hw = hardware_info()

    save_run(
        run_dir=run_dir,
        config=cfg,
        metrics=metrics_payload,
        hardware=hw,
    )

    # ------------------------------------------------------------------
    # Append summary.csv
    # ------------------------------------------------------------------
    flat_row = {
        "run_id": run_id,
        "stage": 1,
        "model": args.model,
        "augmentation": aug_on,
        "seed": cfg.seed,
        "smoke": cfg.smoke,
        "split_scheme": "fixed",
        "device": cfg.device,
        # Flat metric keys
        "bbox_map50": metrics["bbox_map50"],
        "bbox_map50_95": metrics["bbox_map50_95"],
        "segm_map50": metrics["segm_map50"],
        "segm_map50_95": metrics["segm_map50_95"],
        "segm_map75": metrics["segm_map75"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
    }
    append_summary(_RESULTS_BASE, flat_row)

    print(f"[stage1] Done. Artefacts in {run_dir}")
    print(f"[stage1] Summary row appended to {_RESULTS_BASE / 'summary.csv'}")


if __name__ == "__main__":
    main()
