"""KIP Stage 2 — Defect detection on BGAD using the Stage-1 segmentation checkpoint.

Pipeline:
  1. Load the Stage-1 YOLOv11-seg checkpoint (component detection, AP2).
  2. For every BGAD base image: detect + segment the bevel_gear_spindle,
     crop it to the component bbox (mask-gated). This is the Stage-1 -> Stage-2
     hand-off.
  3. Build a PatchCore memory bank from the DEFECT-FREE (good) crops only.
  4. Score every val crop -> per-pixel anomaly map + image-level score.
  5. Evaluate against the BGAD ground-truth defect masks:
       - image-level ROC-AUC (good vs defective)
       - pixel-level ROC-AUC (anomaly map vs defect mask)
     Save metrics + qualitative overlays.

Usage:
    python kip_stage2_bgad.py [--ckpt <best.pt>] [--imgsz 640]
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

# --------------------------------------------------------------------------- #
parser = argparse.ArgumentParser()
parser.add_argument("--root", default=str(Path(__file__).parent))
parser.add_argument("--ckpt", default=None, help="Stage-1 YOLO-seg checkpoint")
parser.add_argument("--imgsz", type=int, default=1024, help="YOLO inference size")
parser.add_argument("--crop", type=int, default=256, help="crop resize for PatchCore")
parser.add_argument("--fg-quantile", type=float, default=0.98,
                    help="image score = this quantile of patch distances")
args = parser.parse_args()

ROOT = Path(args.root).resolve()
BGAD = ROOT / "data" / "BGAD"
CKPT = Path(args.ckpt) if args.ckpt else (
    ROOT / "results/results/yolo_runs/C_synth_pretrain_real_finetune/weights/best.pt")
OUT = ROOT / "results" / "stage2_bgad"
(OUT / "overlays").mkdir(parents=True, exist_ok=True)

DEVICE = "mps" if torch.backends.mps.is_available() else (
    "cuda" if torch.cuda.is_available() else "cpu")
SPINDLE_CLASS = 3  # bevel_gear_spindle in the 9-class Stage-1 model
print(f"root={ROOT}\nckpt={CKPT}\ndevice={DEVICE}")
assert CKPT.exists(), f"Stage-1 checkpoint missing: {CKPT}"

def mask_stem(name: str) -> str:
    m = re.match(r"(.*_isolated(?:_v2)?)_(.+)\.png$", name)
    return m.group(1) if m else name

# --------------------------------------------------------------------------- #
# 1) Stage-1: segment + crop the spindle from each BGAD image
# --------------------------------------------------------------------------- #
from ultralytics import YOLO
yolo = YOLO(str(CKPT))

def crop_spindle(img_path: Path):
    """Return (crop_bgr, bbox xyxy) for the highest-conf spindle, else None."""
    img = cv2.imread(str(img_path))
    res = yolo.predict(source=img, imgsz=args.imgsz, device=DEVICE,
                       verbose=False, conf=0.25)[0]
    if res.boxes is None or len(res.boxes) == 0:
        return None
    cls = res.boxes.cls.cpu().numpy().astype(int)
    conf = res.boxes.conf.cpu().numpy()
    cand = [(i, conf[i]) for i in range(len(cls)) if cls[i] == SPINDLE_CLASS]
    if not cand:  # fall back to best detection of any class
        cand = [(int(conf.argmax()), float(conf.max()))]
    i = max(cand, key=lambda t: t[1])[0]
    x1, y1, x2, y2 = res.boxes.xyxy[i].cpu().numpy().astype(int)
    h, w = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    return img[y1:y2, x1:x2], (x1, y1, x2, y2)

def load_split(split: str):
    items = []
    img_dir = BGAD / split / "base_images"
    msk_dir = BGAD / split / "masks"
    for img in sorted(img_dir.glob("*.jpg")):
        masks = [m for m in msk_dir.glob("*.png") if mask_stem(m.name) == img.stem]
        items.append({"img": img, "masks": masks, "label": int(bool(masks))})
    return items

# --------------------------------------------------------------------------- #
# 2) PatchCore feature extractor (ResNet-18 layer2+layer3), memory bank
# --------------------------------------------------------------------------- #
import torchvision.models as tvm
from torchvision.transforms import functional as TF

net = tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT).eval().to(DEVICE)
_feat = {}
net.layer2.register_forward_hook(lambda m, i, o: _feat.__setitem__("l2", o))
net.layer3.register_forward_hook(lambda m, i, o: _feat.__setitem__("l3", o))

@torch.no_grad()
def embed(crop_bgr):
    """crop -> patch embeddings (P, C) and the feature grid size (gh, gw)."""
    rgb = cv2.cvtColor(cv2.resize(crop_bgr, (args.crop, args.crop)), cv2.COLOR_BGR2RGB)
    t = TF.normalize(TF.to_tensor(rgb), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    net(t.unsqueeze(0).to(DEVICE))
    l2, l3 = _feat["l2"], _feat["l3"]
    gh, gw = l2.shape[-2:]
    l3u = F.interpolate(l3, size=(gh, gw), mode="bilinear", align_corners=False)
    f = torch.cat([l2, l3u], dim=1).squeeze(0)          # (C, gh, gw)
    f = f.permute(1, 2, 0).reshape(-1, f.shape[0])       # (P, C)
    f = F.normalize(f, dim=1)
    return f.cpu(), (gh, gw)

print("\n=== Stage-1 crop extraction ===")
train_items, val_items = load_split("train"), load_split("val")
bank = []
for it in train_items:
    out = crop_spindle(it["img"])
    it["crop"] = None if out is None else out[0]
    it["bbox"] = None if out is None else out[1]
    tag = "GOOD" if it["label"] == 0 else "DEFECT"
    ok = it["crop"] is not None
    print(f"  [train {tag}] {it['img'].name}  crop={'ok' if ok else 'FAIL'}")
    if ok and it["label"] == 0:                          # bank from good only
        emb, _ = embed(it["crop"])
        bank.append(emb)
memory = torch.cat(bank, dim=0)
print(f"memory bank: {tuple(memory.shape)} patches from {len(bank)} good crops")

# --------------------------------------------------------------------------- #
# 3) Score val crops + evaluate
# --------------------------------------------------------------------------- #
@torch.no_grad()
def anomaly_map(crop_bgr):
    emb, (gh, gw) = embed(crop_bgr)
    d = torch.cdist(emb, memory)                         # (P, N)
    score_patch = d.min(dim=1).values.reshape(gh, gw)    # (gh, gw)
    amap = F.interpolate(score_patch[None, None], size=crop_bgr.shape[:2],
                         mode="bilinear", align_corners=False)[0, 0].numpy()
    return amap

def gt_mask_for(it, crop_shape):
    """Union of GT defect masks, cropped to the spindle bbox, resized to crop."""
    x1, y1, x2, y2 = it["bbox"]
    full = None
    for m in it["masks"]:
        arr = cv2.imread(str(m), cv2.IMREAD_GRAYSCALE)
        full = arr if full is None else np.maximum(full, arr)
    if full is None:
        return np.zeros(crop_shape[:2], np.uint8)
    sub = full[y1:y2, x1:x2]
    return (cv2.resize(sub, (crop_shape[1], crop_shape[0])) > 127).astype(np.uint8)

print("\n=== Stage-2 scoring (val) ===")
img_scores, img_labels = [], []
pix_scores, pix_labels = [], []
for it in val_items:
    out = crop_spindle(it["img"])
    if out is None:
        print(f"  [val] {it['img'].name}  no spindle detected -> skip")
        continue
    it["crop"], it["bbox"] = out
    amap = anomaly_map(it["crop"])
    score = float(np.quantile(amap, args.fg_quantile))
    img_scores.append(score); img_labels.append(it["label"])
    tag = "GOOD" if it["label"] == 0 else "DEFECT"
    print(f"  [val {tag:6s}] {it['img'].name}  score={score:.3f}")

    if it["label"] == 1:                                 # pixel eval on defective
        gt = gt_mask_for(it, it["crop"].shape)
        pix_scores.append(amap.ravel()); pix_labels.append(gt.ravel())
        # qualitative overlay
        a = (amap - amap.min()) / (np.ptp(amap) + 1e-9)
        heat = cv2.applyColorMap((a * 255).astype(np.uint8), cv2.COLORMAP_JET)
        ov = cv2.addWeighted(it["crop"], 0.6, heat, 0.4, 0)
        cont, _ = cv2.findContours(gt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(ov, cont, -1, (255, 255, 255), 3)
        cv2.imwrite(str(OUT / "overlays" / f"{it['img'].stem}.jpg"), ov)

metrics = {
    "n_val": len(img_labels),
    "n_good": int(img_labels.count(0)),
    "n_defect": int(img_labels.count(1)),
    "image_auroc": float(roc_auc_score(img_labels, img_scores))
        if len(set(img_labels)) > 1 else None,
    "pixel_auroc": float(roc_auc_score(np.concatenate(pix_labels),
                                       np.concatenate(pix_scores)))
        if pix_labels else None,
    "memory_patches": int(memory.shape[0]),
    "imgsz": args.imgsz, "crop": args.crop, "fg_quantile": args.fg_quantile,
    "ckpt": str(CKPT.relative_to(ROOT)),
}
(OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
print("\n=== RESULTS ===")
print(json.dumps(metrics, indent=2))
print(f"\nsaved -> {OUT/'metrics.json'}  overlays -> {OUT/'overlays'}")
