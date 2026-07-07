# Reproduktion der Ergebnisse

Dieses Dokument beschreibt die vollständige Befehlsfolge zur Reproduktion
aller Ergebnisse auf einem CUDA-fähigen Server.
Lokale Smoke-Tests laufen auf MPS / CPU (macOS).

## Voraussetzungen

```bash
pip install -e .
# Alle Abhängigkeiten sind in requirements-dev.txt gelistet.
```

COCO-Konvertierungen müssen vorhanden sein unter `data/coco_converted/`:
```bash
python scripts/prepare_stage1_coco.py   # falls noch nicht erledigt
```

---

## Stage-1: Komponenten-Segmentierung (YOLO11n-seg vs. Mask2Former)

Acht vollständige GPU-Läufe (CUDA-Server, nicht lokal ausgeführt).
Ausgaben landen ausschließlich unter `results/component_benchmark/`.

### YOLO11n-seg

```bash
# YOLO – Augmentierung AN
python scripts/run_stage1.py \
    --model yolo \
    --aug on \
    --epochs 100 \
    --imgsz 1088 \
    --batch 16 \
    --device cuda:0 \
    --seed 42

# YOLO – Augmentierung AUS (Fairness-Baseline)
python scripts/run_stage1.py \
    --model yolo \
    --aug off \
    --epochs 100 \
    --imgsz 1088 \
    --batch 16 \
    --device cuda:0 \
    --seed 42

# YOLO – Synth-Pretrain-Variante (initialisiert mit vorhandenem Checkpoint):
# Augmentierung AN
python scripts/run_stage1.py \
    --model yolo \
    --aug on \
    --epochs 100 \
    --imgsz 1088 \
    --batch 16 \
    --device cuda:0 \
    --seed 42 \
    --weights results/results/yolo_runs/C_synth_pretrain_real_finetune/weights/best.pt

# YOLO – Synth-Pretrain-Variante, Augmentierung AUS
python scripts/run_stage1.py \
    --model yolo \
    --aug off \
    --epochs 100 \
    --imgsz 1088 \
    --batch 16 \
    --device cuda:0 \
    --seed 42 \
    --weights results/results/yolo_runs/C_synth_pretrain_real_finetune/weights/best.pt
```

### Mask2Former (Swin-T)

```bash
# Mask2Former – Augmentierung AN
python scripts/run_stage1.py \
    --model mask2former \
    --aug on \
    --epochs 100 \
    --batch 8 \
    --lr 1e-4 \
    --freeze-backbone-epochs 20 \
    --device cuda:0 \
    --seed 42

# Mask2Former – Augmentierung AUS (Fairness-Baseline)
python scripts/run_stage1.py \
    --model mask2former \
    --aug off \
    --epochs 100 \
    --batch 8 \
    --lr 1e-4 \
    --freeze-backbone-epochs 20 \
    --device cuda:0 \
    --seed 42

# Mask2Former – Augmentierung AN, kleinere Lernrate für Feinabstimmung
python scripts/run_stage1.py \
    --model mask2former \
    --aug on \
    --epochs 100 \
    --batch 8 \
    --lr 5e-5 \
    --freeze-backbone-epochs 30 \
    --device cuda:0 \
    --seed 42

# Mask2Former – Augmentierung AUS, kleinere Lernrate
python scripts/run_stage1.py \
    --model mask2former \
    --aug off \
    --epochs 100 \
    --batch 8 \
    --lr 5e-5 \
    --freeze-backbone-epochs 30 \
    --device cuda:0 \
    --seed 42
```

### Lokale Smoke-Tests (MPS / CPU)

```bash
# Schnelltest YOLO auf MPS (<=2 Epochen, <=40 Bilder)
.venv/bin/python scripts/run_stage1.py --model yolo --aug on --smoke --device mps

# Schnelltest Mask2Former auf CPU (<=2 Epochen, <=20 Bilder)
.venv/bin/python scripts/run_stage1.py --model mask2former --aug off --smoke --device cpu
```

---

<!-- WP4 ergänzt hier: Stage-2 Befehle, Figuren-Erzeugung -->

## Stufe 2 — Defekterkennung (BGAD): 4 komplementäre Methoden

Manifest (nichtdestruktiv):
```bash
.venv/bin/python scripts/build_manifest.py --bgad data/BGAD \
  --out results/defect_detection/manifest --missing-mask-policy normal
```

Smoke (dieser Rechner, MPS/CPU) — alle vier Methoden, beide Protokolle:
```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
for split in fixed loto; do for m in patchcore padim ae unet; do
  .venv/bin/python scripts/run_stage2.py --method $m --split $split --smoke --device mps
done; done
```

Volltraining (CUDA-Server; NIE als ausgeführt behauptet):
```bash
python scripts/run_stage2.py --method patchcore --split loto --tile 256 --device cuda:0 --seed 42
python scripts/run_stage2.py --method padim     --split loto --tile 256 --device cuda:0 --seed 42
python scripts/run_stage2.py --method ae        --split loto --tile 256 --epochs 200 --device cuda:0 --seed 42
python scripts/run_stage2.py --method unet --aug on --split loto --tile 256 --epochs 300 --loss bce_dice --device cuda:0 --seed 42
```
`--missing-mask-policy {normal|unlabeled|error}` steuert die Annahme "Bild ohne Maske = Gutteil".
LOTO ist primär; `--split fixed` nutzt den vorhandenen train/val-Split (sekundär).

## Gesamter Smoke-Durchlauf
```bash
bash scripts/smoke_all.sh
```
