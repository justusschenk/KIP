"""KIP - Visual Inspection of Angle Grinder Components.

Server training script converted from kip_inspection.ipynb.
Skips all visualisation; saves all models and metrics to disk.

Usage:
    python kip_train.py [--root /path/to/KIP] [--epochs 50] [--variant yolo11n-seg.pt]

Results written to:
    <root>/results/exp_metrics.json
    <root>/results/yolo_runs/{A,B,C}_*/weights/best.pt
    <root>/models/yolo11n_seg.onnx
    <root>/models/patchcore_resnet18.onnx
    <root>/models/patchcore_bank.pt
"""
from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

# ---------------------------------------------------------------------------
# CLI args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="KIP training pipeline")
parser.add_argument("--root", default=str(Path(__file__).parent),
                    help="Project root (default: directory of this script)")
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--variant", default="yolo11n-seg.pt",
                    help="YOLO base weights, e.g. yolo11s-seg.pt")
parser.add_argument("--batch", type=int, default=16)
parser.add_argument("--imgsz", type=int, default=640)
parser.add_argument("--skip-patchcore", action="store_true",
                    help="Skip AP3 PatchCore training (faster run)")
parser.add_argument("--skip-onnx", action="store_true",
                    help="Skip ONNX export (requires onnx/onnxruntime)")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(args.root).resolve()
DATA_DIR     = PROJECT_ROOT / "data"
MODELS_DIR   = PROJECT_ROOT / "models"
RESULTS_DIR  = PROJECT_ROOT / "results"

for d in (MODELS_DIR, RESULTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "synth":   DATA_DIR / "synth_Daten",
    "real_v1": DATA_DIR / "object_segmentation_real_1088",
    "real_v3": DATA_DIR / "object_segmentation_real_v3_1088",
}

CLASS_NAMES = [
    "anti-vibration_handle", "bearing_plate", "bevel_gear_drive",
    "bevel_gear_spindle", "gearbox_housing", "intermediate_gearbox",
    "motor_housing", "shaft", "wheel_guard",
]
NUM_CLASSES = len(CLASS_NAMES)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else (
    "mps" if torch.backends.mps.is_available() else "cpu")

MODEL_VARIANT = args.variant
IMG_SIZE      = args.imgsz
BATCH         = args.batch
EPOCHS        = args.epochs
PROJECT       = str(RESULTS_DIR / "yolo_runs")

print(f"Project root : {PROJECT_ROOT}")
print(f"Device       : {DEVICE}")
print(f"Torch        : {torch.__version__}")
print(f"YOLO variant : {MODEL_VARIANT}  epochs={EPOCHS}  imgsz={IMG_SIZE}  batch={BATCH}")

for name, p in DATASETS.items():
    print(f"  dataset {name:8s} -> exists={p.exists()}  ({p})")

# ---------------------------------------------------------------------------
# AP1 - Data preparation
# ---------------------------------------------------------------------------
TOOL_RE = re.compile(r"(tool\d+)_")

def _tool_of(name: str) -> str | None:
    m = TOOL_RE.match(name)
    return m.group(1) if m else None


def create_tool_based_split(dataset_path: Path,
                             test_tools: tuple[str, ...] = ("tool10",)) -> tuple[int, int]:
    img_val  = dataset_path / "images" / "val"
    lbl_val  = dataset_path / "labels" / "val"
    img_test = dataset_path / "images" / "test"
    lbl_test = dataset_path / "labels" / "test"
    img_test.mkdir(parents=True, exist_ok=True)
    lbl_test.mkdir(parents=True, exist_ok=True)

    test_set = set(test_tools)
    moved = 0
    for img in sorted(img_val.glob("*")):
        if _tool_of(img.name) in test_set:
            shutil.move(str(img), str(img_test / img.name))
            lbl = lbl_val / (img.stem + ".txt")
            if lbl.exists():
                shutil.move(str(lbl), str(lbl_test / lbl.name))
            moved += 1

    for cache in (lbl_val.parent / "val.cache", lbl_val.parent / "test.cache"):
        if cache.exists():
            cache.unlink()

    val_remaining = sum(1 for _ in img_val.glob("*"))
    val_tools_actual  = sorted({_tool_of(p.name) for p in img_val.glob("*")} - {None})
    test_tools_actual = sorted({_tool_of(p.name) for p in img_test.glob("*")} - {None})
    print(f"  [{dataset_path.name}] moved {moved} -> test/  "
          f"val={val_remaining} {val_tools_actual}  "
          f"test={sum(1 for _ in img_test.glob('*'))} {test_tools_actual}")
    return moved, val_remaining


def write_eval_yaml(dataset_path: Path, out_path: Path) -> Path:
    yaml_text = (
        f"path: {dataset_path}\n"
        f"train: images/train\n"
        f"val: images/test\n"
        f"nc: {NUM_CLASSES}\n"
        "names:\n"
        + "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASS_NAMES))
    )
    out_path.write_text(yaml_text)
    return out_path


print("\n=== AP1: data split ===")
for name in ("real_v1", "real_v3"):
    if DATASETS[name].exists():
        create_tool_based_split(DATASETS[name], test_tools=("tool10",))

EVAL_YAML_REAL = write_eval_yaml(DATASETS["real_v3"],
                                  RESULTS_DIR / "eval_real_v3_test.yaml")
print(f"OOS eval YAML -> {EVAL_YAML_REAL}")

# ---------------------------------------------------------------------------
# AP2 - YOLO segmentation experiments
# ---------------------------------------------------------------------------
from ultralytics import YOLO


def train_yolo(data_yaml: Path, model_variant: str, epochs: int,
               imgsz: int = 640, batch: int = 16, name: str = "exp",
               weights: str | None = None) -> YOLO:
    init = weights or model_variant
    model = YOLO(init)
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=DEVICE,
        project=PROJECT,
        name=name,
        exist_ok=True,
        seed=SEED,
        verbose=True,
    )
    return model


def evaluate_yolo(model: YOLO, eval_yaml: Path, name: str = "") -> dict:
    res = model.val(data=str(eval_yaml), imgsz=IMG_SIZE, batch=BATCH,
                    device=DEVICE, verbose=False, split="val")
    box = res.box
    metrics = {
        "mAP50":     float(box.map50),
        "mAP50_95":  float(box.map),
        "precision": float(box.mp),
        "recall":    float(box.mr),
        "f1_macro":  float((2 * box.mp * box.mr) / (box.mp + box.mr + 1e-9)),
    }
    print(f"  [{name}] {metrics}")
    return metrics


print("\n=== AP2: YOLO training ===")
all_metrics: dict[str, dict] = {}

# --- Exp A: synthetic only ---
synth_yaml = DATASETS["synth"] / "data.yaml"
if synth_yaml.exists():
    print("\n-- Exp A: synth only --")
    model_a = train_yolo(synth_yaml, MODEL_VARIANT, EPOCHS, IMG_SIZE, BATCH,
                         name="A_synth_only")
    all_metrics["A_synth_only"] = evaluate_yolo(model_a, EVAL_YAML_REAL, "A")
    # save best weights path for reference
    best_a = Path(PROJECT) / "A_synth_only" / "weights" / "best.pt"
    print(f"  best.pt -> {best_a}  exists={best_a.exists()}")
else:
    print(f"  synth dataset YAML missing at {synth_yaml}, skipping Exp A")
    model_a = None

# --- Exp B: real v3 only ---
real_yaml = DATASETS["real_v3"] / "data.yaml"
if real_yaml.exists():
    print("\n-- Exp B: real v3 only --")
    model_b = train_yolo(real_yaml, MODEL_VARIANT, EPOCHS, IMG_SIZE, BATCH,
                         name="B_real_only")
    all_metrics["B_real_only"] = evaluate_yolo(model_b, EVAL_YAML_REAL, "B")
    best_b = Path(PROJECT) / "B_real_only" / "weights" / "best.pt"
    print(f"  best.pt -> {best_b}  exists={best_b.exists()}")
else:
    print(f"  real_v3 dataset YAML missing at {real_yaml}, skipping Exp B")
    model_b = None

# --- Exp C: synth pretrain -> real fine-tune ---
if real_yaml.exists():
    print("\n-- Exp C: synth pretrain -> real fine-tune --")
    synth_weights = Path(PROJECT) / "A_synth_only" / "weights" / "best.pt"
    model_c = train_yolo(
        real_yaml, MODEL_VARIANT, EPOCHS, IMG_SIZE, BATCH,
        name="C_synth_pretrain_real_finetune",
        weights=str(synth_weights) if synth_weights.exists() else None,
    )
    all_metrics["C_transfer"] = evaluate_yolo(model_c, EVAL_YAML_REAL, "C")
    best_c = Path(PROJECT) / "C_synth_pretrain_real_finetune" / "weights" / "best.pt"
    print(f"  best.pt -> {best_c}  exists={best_c.exists()}")
else:
    model_c = None

# Persist metrics summary
metrics_path = RESULTS_DIR / "exp_metrics.json"
metrics_path.write_text(json.dumps(all_metrics, indent=2))
print(f"\nMetrics saved -> {metrics_path}")

# ---------------------------------------------------------------------------
# AP3 - PatchCore (optional)
# ---------------------------------------------------------------------------
if not args.skip_patchcore:
    print("\n=== AP3: PatchCore ===")

    def extract_component_crops(yolo_model: YOLO, image_dir: Path, class_id: int,
                                 output_dir: Path, padding: int = 10,
                                 imgsz: int = 640, max_crops: int = 200) -> int:
        output_dir.mkdir(parents=True, exist_ok=True)
        images = sorted(image_dir.glob("*"))
        saved = 0
        for img_path in tqdm(images, desc=f"crops/{CLASS_NAMES[class_id]}"):
            if saved >= max_crops:
                break
            res = yolo_model.predict(source=str(img_path), imgsz=imgsz,
                                     device=DEVICE, verbose=False)[0]
            if res.boxes is None or len(res.boxes) == 0:
                continue
            img = cv2.imread(str(img_path))
            h, w = img.shape[:2]
            for i, cls in enumerate(res.boxes.cls.cpu().numpy().astype(int)):
                if cls != class_id:
                    continue
                x1, y1, x2, y2 = res.boxes.xyxy[i].cpu().numpy().astype(int)
                x1 = max(0, x1 - padding); y1 = max(0, y1 - padding)
                x2 = min(w, x2 + padding); y2 = min(h, y2 + padding)
                crop = img[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                cv2.imwrite(str(output_dir / f"{img_path.stem}_{i}.png"), crop)
                saved += 1
                if saved >= max_crops:
                    break
        return saved

    def _train_patchcore_fallback(normal_crops_dir: Path, backbone: str = "resnet18"):
        import torchvision.models as tvm
        from torchvision.transforms import functional as TF

        if backbone == "resnet18":
            net = tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT)
        else:
            net = tvm.wide_resnet50_2(weights=tvm.Wide_ResNet50_2_Weights.DEFAULT)
        extractor = torch.nn.Sequential(*list(net.children())[:7]).eval().to(DEVICE)

        feats = []
        for img_path in sorted(normal_crops_dir.glob("*.png")):
            img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))
            t = TF.to_tensor(img)
            t = TF.normalize(t, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            with torch.no_grad():
                f = extractor(t.unsqueeze(0).to(DEVICE)).squeeze(0)
            f = f.permute(1, 2, 0).reshape(-1, f.shape[0]).cpu()
            feats.append(f)
        if not feats:
            raise RuntimeError(f"No crops in {normal_crops_dir}")
        bank = torch.cat(feats, dim=0)
        idx = torch.randperm(bank.shape[0])[: max(1, bank.shape[0] // 10)]
        return ("fallback", {"bank": bank[idx], "extractor": extractor})

    def train_patchcore(normal_crops_dir: Path, backbone: str = "resnet18",
                        layers: tuple[str, ...] = ("layer2", "layer3"),
                        coreset_ratio: float = 0.1):
        try:
            from anomalib.models.image import Patchcore
            from anomalib.data.image.folder import Folder
            from anomalib.engine import Engine

            datamodule = Folder(
                name="kip_normal",
                root=str(normal_crops_dir.parent),
                normal_dir=normal_crops_dir.name,
                image_size=(224, 224),
                train_batch_size=8,
                eval_batch_size=8,
                num_workers=2,
            )
            model = Patchcore(backbone=backbone, layers=list(layers),
                              coreset_sampling_ratio=coreset_ratio)
            engine = Engine(accelerator="auto", devices=1, max_epochs=1)
            engine.fit(model=model, datamodule=datamodule)
            return ("anomalib", model, engine)
        except Exception as exc:
            print(f"  anomalib unavailable ({str(exc)[:80]}); using fallback.")
            return _train_patchcore_fallback(normal_crops_dir, backbone)

    def export_patchcore(patchcore_model, out_dir: Path) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        backend, payload = patchcore_model[0], patchcore_model[1]
        if backend == "fallback":
            extractor = payload["extractor"]
            bank = payload["bank"]
            dummy = torch.randn(1, 3, 224, 224).to(DEVICE)
            onnx_path = out_dir / "patchcore_resnet18.onnx"
            torch.onnx.export(extractor, dummy, str(onnx_path),
                              input_names=["input"], output_names=["features"],
                              opset_version=12,
                              dynamic_axes={"input": {0: "N"}, "features": {0: "N"}})
            torch.save(bank, out_dir / "patchcore_bank.pt")
            print(f"  PatchCore ONNX -> {onnx_path}")
            print(f"  PatchCore bank -> {out_dir / 'patchcore_bank.pt'}")
            return onnx_path
        ts_path = out_dir / "patchcore_anomalib.ts"
        try:
            torch.jit.save(torch.jit.script(payload), str(ts_path))
            print(f"  PatchCore TorchScript -> {ts_path}")
        except Exception as exc:
            print(f"  Could not TorchScript-export anomalib model: {exc}")
        return ts_path

    yolo_for_crops = model_c or model_b or model_a
    if yolo_for_crops is not None and (DATASETS["real_v3"] / "images" / "train").exists():
        crops_dir = DATA_DIR / "patchcore_normal" / "motor_housing"
        n = extract_component_crops(
            yolo_for_crops,
            DATASETS["real_v3"] / "images" / "train",
            class_id=CLASS_NAMES.index("motor_housing"),
            output_dir=crops_dir,
            padding=10, max_crops=80,
        )
        print(f"  saved {n} crops -> {crops_dir}")
        if n > 0:
            pc = train_patchcore(crops_dir, backbone="resnet18")
            export_patchcore(pc, MODELS_DIR)
        else:
            print("  No crops extracted, skipping PatchCore training.")
    else:
        print("  No trained YOLO or no real_v3 train images; skipping PatchCore.")
else:
    print("\n=== AP3: PatchCore skipped (--skip-patchcore) ===")

# ---------------------------------------------------------------------------
# AP4 - ONNX export
# ---------------------------------------------------------------------------
if not args.skip_onnx:
    print("\n=== AP4: ONNX export ===")

    def export_yolo_onnx(model: YOLO, out_path: Path, imgsz: int = 640) -> Path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        onnx_path = model.export(format="onnx", imgsz=imgsz, half=True,
                                 simplify=True, opset=12)
        onnx_path = Path(onnx_path)
        if onnx_path.resolve() != out_path.resolve():
            shutil.copy2(onnx_path, out_path)
        print(f"  YOLO ONNX -> {out_path}")
        return out_path

    src_model = model_c or model_b or model_a
    if src_model is not None:
        yolo_onnx = export_yolo_onnx(src_model, MODELS_DIR / "yolo11n_seg.onnx",
                                     imgsz=IMG_SIZE)
    else:
        print("  No trained model available, skipping ONNX export.")
else:
    print("\n=== AP4: ONNX export skipped (--skip-onnx) ===")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n=== Done ===")
print(f"Results dir  : {RESULTS_DIR}")
print(f"Models dir   : {MODELS_DIR}")
print("\nFiles written:")
for p in sorted(list(RESULTS_DIR.rglob("*")) + list(MODELS_DIR.rglob("*"))):
    if p.is_file():
        size_mb = p.stat().st_size / 1e6
        print(f"  {p.relative_to(PROJECT_ROOT)}  ({size_mb:.1f} MB)")
