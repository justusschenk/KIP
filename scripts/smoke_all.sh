#!/usr/bin/env bash
# End-to-end CPU/MPS smoke suite for the KIP two-stage model benchmark.
# Proves the whole pipeline runs and produces REAL (smoke-flagged) metrics.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export PYTORCH_ENABLE_MPS_FALLBACK=1

echo "== unit tests =="
.venv/bin/pytest tests/ -q

echo "== manifest =="
$PY scripts/build_manifest.py --bgad data/BGAD \
    --out results/defect_detection/manifest --missing-mask-policy normal

echo "== stage 1 (component segmentation) =="
$PY scripts/run_stage1.py --model yolo        --aug on  --smoke --device mps
$PY scripts/run_stage1.py --model mask2former --aug off --smoke --device cpu

echo "== stage 2 (defect detection) — 4 methods, LOTO + fixed =="
for split in fixed loto; do
  for m in patchcore padim ae unet; do
    $PY scripts/run_stage2.py --method "$m" --split "$split" --smoke --device mps
  done
done

echo "== figures =="
$PY scripts/make_figures.py all

echo "== DONE — see results/component_benchmark, results/defect_detection, results/figures =="
