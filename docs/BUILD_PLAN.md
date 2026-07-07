# KIP Model-Benchmark — Build Plan (single source of truth)

This plan governs a scientific-paper-grade model comparison in two stages.
All coding agents MUST read this file and follow it exactly.

## 0. Ground rules for all coders
- Python: `/Users/justusschenk/Projects/KIP/.venv/bin/python` (3.14, torch 2.11, MPS). Never assume CUDA locally.
- Dependencies are ALREADY INSTALLED and import-verified on this py3.14 venv:
  `transformers 5.13`, `timm 1.0.27`, `segmentation-models-pytorch 0.5`, `pycocotools`, `pytest 9.1`, `scikit-image 0.26`.
  Mask2Former classes import and smp `Unet('resnet18')` does a forward pass. Do NOT re-litigate deps; still list them in `requirements-dev.txt`.
- NEVER write into `results/results/*` or `results/stage2_bgad/*`. New outputs only under `results/component_benchmark/` and `results/defect_detection/`.
- Every run writes REAL computed numbers; smoke runs set `"smoke": true` in metrics.json. Never fabricate full-scale results.
- Class order (fixed, matches all `data.yaml`): `0 anti-vibration_handle, 1 bearing_plate, 2 bevel_gear_drive, 3 bevel_gear_spindle, 4 gearbox_housing, 5 intermediate_gearbox, 6 motor_housing, 7 shaft, 8 wheel_guard`. Spindle class id = 3.
- Reuse logic from `kip_stage2_bgad.py` (mask_stem regex, crop_spindle, embed) and `kip_train.py` (split/eval patterns) — fold in, don't duplicate ad hoc.

## Verified data facts
- train/masks ≡ val/masks (identical 13 files); several masks reference images living in the *other* split
  (val images tool02_0011, tool03_0003, tool08_0014, tool09_0012, tool10_0008_v2 have masks present in train/masks).
- Pooled BGAD = 7 tools {02,03,08,09,10,97,99}, 19 images (8 good / 11 defect). tool03 = 1 image (defect-only);
  tool97/99 are good-only. `_v2` token is part of the image stem (`tool10_..._0008_isolated_v2.jpg` ↔ `..._isolated_v2_polishing_wear.png`).
- real_v3 = 771 train / 101 val / 90 test. Stage-1 checkpoint C best.pt exists at
  `results/results/yolo_runs/C_synth_pretrain_real_finetune/weights/best.pt`.

## 1. Module/file tree
```
kip/
  __init__.py            # version, CLASS_NAMES, SPINDLE_CLASS=3
  config.py              # dataclass configs (Stage1Config, Stage2Config, TilingConfig), YAML load/dump, seed_everything()
  hardware.py            # hardware_info() -> dict, device auto-pick
  data/
    bgad_manifest.py     # manifest build/validate/save: dedup, pair-by-split, union, missing-mask policy
    tiling.py            # overlapping tiles, bg-tile filter, stitch
    splits.py            # LOTO / GroupKFold / fixed by tool_id + leakage assertion
    yolo_to_coco.py      # YOLO-seg -> COCO instance JSON
    datasets.py          # TileDataset, SupervisedTileDataset, crop cache
    augment.py           # FULL: stage1 aug on/off hyp dicts + stage2 mask-consistent albumentations pipeline
  stage1/
    crop.py              # Stage-1 ckpt -> spindle crop (port of kip_stage2_bgad.crop_spindle, + fallback + cache)
    yolo_trainer.py      # YOLO11n-seg train/predict-to-COCO
    mask2former_trainer.py  # HF Mask2Former Swin-T fine-tune (plain torch) + predict-to-COCO
    evaluator.py         # single fair pycocotools evaluator (bbox+segm), per-class AP
  stage2/
    base.py              # AnomalyMethod ABC, registry, per-fold score normalization
    patchcore.py; padim.py; autoencoder.py; unet.py; runner.py
  metrics/
    detection.py         # COCOeval wrappers
    anomaly.py           # image/pixel AUROC, AUPRO, best-F1, Dice/IoU, degenerate guards
  reporting/
    results_io.py        # run-dir, schema writers, git hash
    figures.py           # comparison bars, ROC/PRO, overlays, per-class AP
scripts/
  build_manifest.py; prepare_stage1_coco.py; run_stage1.py; run_stage2.py; make_figures.py; smoke_all.sh
tests/
  conftest.py; test_manifest.py; test_tiling.py; test_splits.py; test_yolo_to_coco.py; test_metrics.py; test_stage2_contract.py
docs/REPRODUKTION.md     # GERMAN reproduction guide
requirements-dev.txt     # updated
pyproject.toml           # package=kip, pytest config
```

## 2. Key interfaces (implement exactly)

### 2.1 Manifest (`kip/data/bgad_manifest.py`)
```python
MASK_STEM_RE = re.compile(r"(.*_isolated(?:_v2)?)_(.+)\.png$")  # g1=image stem, g2=defect type
def normalize_defect_type(raw: str) -> tuple[str, bool]:   # 'v2_polishing_wear'->('polishing_wear', True)
def build_manifest(bgad_root, missing_mask_policy: Literal["normal","unlabeled","error"]="normal") -> pd.DataFrame
    # cols: image, tool_id, split, defect_status(good|defect|unlabeled), defect_types(';'), mask_paths(';'), width, height
    # dedup identical mask filenames across split dirs; pair mask->image ONLY if image exists in same split.
def union_masks(mask_paths, size_hw) -> np.ndarray            # uint8 {0,1}
def validate_manifest(df, bgad_root) -> list[str]            # mask readable, values in {0,255}, size match, empty-mask, tool parse, dup images; raises ManifestError on fatal
def save_manifest(df, out_dir) -> Path                       # manifest.csv + manifest_meta.json (policy, sha256, counts)
```

### 2.2 Tiling (`kip/data/tiling.py`)
```python
@dataclass
class Tile: image_id: str; x0:int; y0:int; w:int; h:int; fg_fraction: float
def compute_grid(h, w, tile=576, overlap=0.25) -> list[tuple[int,int]]
def is_background_tile(tile_bgr, white_thresh=240, white_frac=0.95) -> bool
def tile_image(img, mask|None, cfg: TilingConfig) -> list[tuple[Tile, np.ndarray, np.ndarray|None]]  # drops bg
def stitch(tiles, full_hw, mode: Literal["mean","max"]="mean", fill_value=0.0) -> np.ndarray  # weighted overlap blend
```

### 2.3 Splits (`kip/data/splits.py`)
```python
@dataclass
class Fold: name:str; train_idx:np.ndarray; test_idx:np.ndarray; held_out_tools:list[str]
def leave_one_tool_out(manifest) -> list[Fold]
def group_kfold(manifest, n_splits, seed) -> list[Fold]
def fixed_split(manifest) -> Fold
def assert_no_tool_leakage(fold, manifest) -> None   # raises; called by runner AND tested
```
Split on the manifest (full images) strictly BEFORE tiling; tiles inherit fold via image_id.

### 2.4 YOLO->COCO (`kip/data/yolo_to_coco.py`)
```python
def convert_split(images_dir, labels_dir, class_names, out_json, start_ids=(1,1)) -> dict
    # polygons denormalized to abs px, bbox from extent, area via cv2 raster, iscrowd=0; images w/ no label -> zero anns; skip <3-pt polygons
def parse_yolo_seg_line(line, w, h) -> tuple[int, np.ndarray]   # (cls, Nx2 abs)
```

### 2.5 Stage-1 (`kip/stage1/`)
```python
# crop.py (WP1)
def crop_spindle(img_bgr, yolo_model, imgsz=1024, conf=0.25, fallback: Literal["full","best-box","skip"]="best-box") -> tuple[np.ndarray, tuple]|None

class YoloSegTrainer:
    def train(self) -> Path                       # aug off = mosaic/mixup/hsv/flip/scale/translate/erasing all 0
    def predict_to_coco(self, ckpt, coco_gt_json, images_dir, out_json) -> Path
class Mask2FormerTrainer:                          # facebook/mask2former-swin-tiny-coco-instance
    def train(self) -> Path                        # plain torch loop, freeze backbone initially, albumentations aug when on
    def predict_to_coco(self, ckpt_dir, coco_gt_json, images_dir, out_json) -> Path
def evaluate_coco(gt_json, pred_json, class_names) -> dict   # SINGLE evaluator; bbox+segm; per-class segm AP
```
Fairness: identical image lists (same COCO conversion), same eval code, same test split (real_v3/test), matched aug spec, documented resolution per model. GT-as-pred sanity -> segm_map50 ≈ 1.0.

### 2.6 Stage-2 (`kip/stage2/`)
```python
class AnomalyMethod(ABC):
    name: str
    def fit(self, good_tiles) -> None
    def score(self, tile) -> tuple[np.ndarray, float]       # (amap HxW f32, image_score)
    def score_full_image(self, img, tiles_cfg) -> tuple[np.ndarray, float]   # tile->score->stitch (base default)
class PatchCore(AnomalyMethod)   # resnet18 l2+l3 bank, coreset if bank>10k, cdist on CPU
class PaDiM(AnomalyMethod)       # d_reduced=100 random dims, cov shrinkage eps*I, Mahalanobis
class ConvAE(AnomalyMethod)      # train on good tiles; score = smoothed recon error (L2 [+SSIM])
class UNetSupervised:            # smp.Unet('resnet18'); loss bce_dice|focal; needs labels
    def fit_supervised(self, train_ds, val_ds) -> Path
    def score(self, tile) -> tuple[np.ndarray, float]       # sigmoid map; image_score = map.max()

# runner.py
def run_stage2(method, split, aug, cfg, smoke) -> Path
def normalize_fold_scores(scores, good_ref_scores) -> np.ndarray  # (s-median(good))/(IQR(good)+eps)
```

### 2.7 Metrics (`kip/metrics/anomaly.py`)
```python
def image_auroc(labels, scores) -> float|None            # None if single class
def pixel_auroc(gt, amap) -> float|None
def aupro(gt_masks, amaps, fpr_limit=0.3) -> float|None  # scipy.ndimage.label regions
def best_f1(labels, scores) -> tuple[float,float]        # (f1, threshold)
def dice_iou(gt, pred_bin) -> dict                       # empty-vs-empty := 1.0
```
`kip/metrics/detection.py`: `coco_summary(cocoeval) -> dict`.

### 2.8 Reporting (`kip/reporting/`)
```python
def create_run_dir(base, run_name) -> Path               # base/<run>_<YYYYmmdd_HHMMSS>/
def save_run(run_dir, config, metrics, manifest_path|None, curves|None, hardware) -> None
def append_summary(base, flat_row) -> None               # base/summary.csv stable columns
def hardware_info() -> dict                              # kip/hardware.py
# figures.py: fig_stage1_comparison, fig_stage1_per_class, fig_stage2_comparison, fig_roc_curves, fig_overlays
```

## 3. Results-artifact schema
Run dir (both stages):
```
<base>/<run>/ config.yaml  metrics.json  hardware.json  [manifest.csv]  curves/  predictions/  figures/
<base>/summary.csv   # one flat row per run, append-only
```
`metrics.json` envelope:
```json
{"schema_version":"1.0","run_id":"...","stage":1,"model":"...","augmentation":true,"seed":42,"smoke":false,
 "device":"mps","git_commit":"...","timestamp_utc":"...",
 "split":{"scheme":"fixed|loto|gkf","n_folds":7,"test_set":"real_v3/test"},
 "dataset":{"n_train":0,"n_val":0,"n_test":0,"n_good":0,"n_defect":0,"n_tiles_train":0,"n_tiles_test":0,
            "missing_mask_policy":"normal","crop_failures":0},
 "runtime":{"train_seconds":0.0,"infer_ms_per_image":0.0},"metrics":{}}
```
Stage-1 `metrics`: `bbox_map50, bbox_map50_95, segm_map50, segm_map50_95, segm_map75, precision, recall, per_class:{<name>:{segm_ap50,segm_ap50_95}}`.
Stage-2 `metrics`: `pooled:{image_auroc,image_f1,image_f1_threshold,pixel_auroc,pixel_aupro,pixel_f1,dice,iou}` (dice/iou unet-only, else null) and `per_fold:[{fold,held_out_tools,n_test,n_defect,image_auroc|null,pixel_auroc|null,pixel_aupro|null,dice|null}]`. Pooled image metrics on fold-normalized scores; pooled pixel metrics on concatenated pixels of ALL test images.
`summary.csv` cols: `run_id,stage,model,augmentation,seed,smoke,split_scheme,device,` + flattened `metric.<key>`.

## 4. Work packages (ordered; WP2 ∥ WP3 after WP1)

### WP1 — Foundation (blocks WP2+WP3)
Deliverables: `pyproject.toml`, `kip/__init__.py`, `config.py`, `hardware.py`, ALL of `kip/data/` (incl. FULL `augment.py` with BOTH stage1 hyp dicts and stage2 mask-consistent pipeline), **`kip/stage1/crop.py`** (port from prototype), ALL of `kip/metrics/`, `kip/reporting/results_io.py`, `scripts/build_manifest.py`, `scripts/prepare_stage1_coco.py`, tests for data+metrics, updated `requirements-dev.txt`.
Acceptance:
- `pytest tests/ -q` green for test_manifest/test_tiling/test_splits/test_yolo_to_coco/test_metrics, with fixtures reproducing: duplicated masks across splits, cross-split mask, `_v2` stem, multi-mask image, empty label file.
- `build_manifest.py --bgad data/BGAD --out results/defect_detection/manifest --missing-mask-policy normal` on REAL data reports: 19 images, 7 tools, 8 good / 11 defect, per-split pairings (train 6 defect imgs, val 5), 0 orphan pairings.
- `prepare_stage1_coco.py` converts real_v3 train/val/test; output validates with `pycocotools.coco.COCO`.

### WP2 — Stage-1 benchmark: YOLO vs Mask2Former (depends WP1)
Deliverables: `kip/stage1/yolo_trainer.py`, `mask2former_trainer.py`, `evaluator.py`, `scripts/run_stage1.py`, stage-1 figure hooks. (crop.py + augment.py come from WP1 — consume, don't redefine.)
Acceptance:
- Smoke green: `run_stage1.py --model yolo --aug on --smoke` (≤2 epochs, imgsz 320, ≤40 imgs, MPS) and `--model mask2former --aug off --smoke` (≤2 epochs, ≤20 imgs, batch 2) — each writes schema-conformant run dir under `results/component_benchmark/` + appends summary.csv.
- Both via same `evaluate_coco()`; GT-as-pred -> segm_map50 ≈ 1.0.
- 8 full-run GPU commands in `docs/REPRODUKTION.md`.
- `results/results/*` untouched (assert output base not under results/results).

### WP3 — Stage-2 defect benchmark: 4 methods (depends WP1; uses crop.py + augment.py from WP1)
Deliverables: ALL of `kip/stage2/`, `scripts/run_stage2.py`, `tests/test_stage2_contract.py`.
Acceptance:
- `pytest tests/test_stage2_contract.py -q` green: every method fits on 4 random 576² good tiles, returns correct-shaped `(amap, score)`, deterministic under seed (CPU).
- Smoke green on real data all four: `run_stage2.py --method {patchcore|padim|ae|unet} --split loto --smoke` (ae/unet ≤3 epochs, tile 256). Per-fold + pooled metrics; single-class folds -> per-fold image_auroc null but pooled image_auroc real; `assert_no_tool_leakage` every fold.
- `--split fixed` runs; `--aug on|off` for unet. PatchCore pooled image-AUROC on `--split fixed` within ±0.15 of prototype 0.867.

### WP4 — Figures, docs, integration (depends WP2+WP3)
Deliverables: `kip/reporting/figures.py`, `scripts/make_figures.py`, `scripts/smoke_all.sh`, `docs/REPRODUKTION.md` (GERMAN), requirements freeze notes.
Acceptance:
- `bash scripts/smoke_all.sh` runs manifest -> both stage-1 smokes -> all four stage-2 smokes -> make_figures green end-to-end.
- `make_figures.py` produces ≥4 figures from real smoke summary.csv.
- `pytest -q` fully green; REPRODUKTION.md commands verified.

## 5. Dependencies + commands
`requirements-dev.txt` additions: `transformers>=5.0`, `segmentation-models-pytorch>=0.5`, `timm>=1.0`, `pycocotools>=2.0.8`, `pytest>=8.0`, `scikit-image>=0.25`, `pyyaml>=6.0`.
Smoke (this machine):
```bash
.venv/bin/pip install -e . && .venv/bin/pytest tests/ -q
.venv/bin/python scripts/build_manifest.py --bgad data/BGAD --out results/defect_detection/manifest --missing-mask-policy normal
.venv/bin/python scripts/run_stage1.py --model yolo        --aug on  --smoke --device mps
.venv/bin/python scripts/run_stage1.py --model mask2former --aug off --smoke --device cpu
for m in patchcore padim ae unet; do .venv/bin/python scripts/run_stage2.py --method $m --split loto --smoke --device mps; done
.venv/bin/python scripts/make_figures.py --stage all
```
Full runs (documented for CUDA server; NEVER claimed as executed):
```bash
python scripts/run_stage1.py --model yolo        --aug {on,off} --epochs 100 --imgsz 1088 --batch 16 --device cuda:0 --seed 42
python scripts/run_stage1.py --model mask2former --aug {on,off} --epochs 100 --batch 8 --lr 1e-4 --freeze-backbone-epochs 20 --device cuda:0 --seed 42
python scripts/run_stage2.py --method {patchcore,padim,ae} --split loto --device cuda:0 --seed 42
python scripts/run_stage2.py --method unet --aug {on,off} --split loto --epochs 200 --batch 8 --loss bce_dice --device cuda:0 --seed 42
```

## 6. Risks / edge cases (MUST handle)
1. BGAD mask quirks: identical masks in train/masks & val/masks; masks whose base image is in the OTHER split. Pair strictly within split; pooled 19-image manifest is the primary LOTO object.
2. `_v2` stem: `(.*_isolated(?:_v2)?)_(.+)\.png`; normalize `v2_polishing_wear`->`polishing_wear`+variant. Types are metadata only — NO multi-class defect claims.
3. Degenerate LOTO folds: tool03=1 defect-only image; tool97/99 good-only -> per-fold image-AUROC null (never 0.0); pool image scores AFTER per-fold normalization vs good-train.
4. Pixel metrics: good test images contribute all-negative pixels to pooled pixel-AUROC (MVTec convention); AUPRO over defect images only; Dice empty-vs-empty:=1.0, empty-GT-vs-nonempty:=0.0.
5. Tiling 5472×3648 white-bg isolated images: drop tiles ≥95% near-white; assert kept tiles cover ≥99% of GT-positive pixels else lower bg threshold; stitch mean-weighted; uncovered regions get fold min score.
6. Tiny supervised U-Net (~10 imgs/fold): oversample positive tiles, pos_weight/Dice for imbalance, early-stop on val Dice; report per-fold spread.
7. PaDiM singular cov: shrinkage Σ+εI (ε≈0.01) mandatory, 100 random dims. PatchCore: skip coreset when bank<10k; cdist on CPU.
8. MPS: no float64; set `PYTORCH_ENABLE_MPS_FALLBACK=1`; ultralytics AMP off on MPS; `torch.use_deterministic_algorithms(warn_only=True)`; MPS numbers not bit-identical to CUDA.
9. Mask2Former tiny data: freeze Swin backbone initially; lr 1e-4/1e-5; grad clip; processor needs per-instance binary masks + class labels (instance, not semantic); handle zero-instance images; overfitting on 771 imgs is a paper finding, not a bug.
10. Fairness (stage 1): never compare ultralytics `model.val()` vs pycocotools; both go through `evaluate_coco()`; identical image lists; YOLO aug-off zeros mosaic/mixup/hsv/flips/scale/translate/erasing.
11. Crop failures: ckpt may miss spindle -> `--crop-fallback full|best-box|skip`; count in `dataset.crop_failures`.
12. Smoke honesty: smoke computes REAL metrics with `"smoke":true`; figures label smoke sources; writeup uses full-run numbers only for claims.
13. Path safety: hard-assert output roots are `results/component_benchmark|defect_detection`; never touch `results/results/`, `results/stage2_bgad/`, checkpoints; cache COCO under `data/coco_converted/` (new dir), originals read-only.
