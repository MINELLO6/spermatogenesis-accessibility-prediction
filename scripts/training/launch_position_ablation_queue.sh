#!/usr/bin/env bash
set -euo pipefail

ROOT="${ACCESSIBILITY_RUN_ROOT:-$PWD/runs}"
read -r -a GPUS <<< "${GPU_IDS:-0}"
SHAPE_OUT="$ROOT/shape_motif_nopos_equalce5"
MAG_OUT="$ROOT/dense_motif_nopos5"
LOG_DIR="$ROOT/position_ablation_queue/logs"
mkdir -p "$LOG_DIR"

run_fold() {
  local gpu="$1"
  local fold="$2"
  CUDA_VISIBLE_DEVICES="$gpu" python -m scripts.training.train_shape_shared_trunk \
    --modality motif_nopos --fold "$fold" --objective equal_ce \
    --epochs 30 --patience 5 --batch-size 1024 --accum-steps 4 \
    --lr 0.001 --p-threshold 0.001 --rc-prob 0.0 \
    --output-root "$SHAPE_OUT" \
    >"$LOG_DIR/shape_nopos_fold$((fold + 1)).log" 2>&1

  CUDA_VISIBLE_DEVICES="$gpu" python -m scripts.training.train_dense_motif_magnitude \
    --variant conv_nopos --fold "$fold" --epochs 25 --patience 5 \
    --batch-size 1024 --accum-steps 2 --lr 0.0003 --seed-offset 100 \
    --rc-prob 0.0 --output-root "$MAG_OUT" \
    >"$LOG_DIR/magnitude_nopos_fold$((fold + 1)).log" 2>&1
}

worker() {
  local slot="$1" fold
  for ((fold=slot; fold<5; fold+=${#GPUS[@]})); do
    run_fold "${GPUS[$slot]}" "$fold"
  done
}

pids=()
for slot in "${!GPUS[@]}"; do
  worker "$slot" &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
if ((status)); then
  echo "At least one fold failed; inspect $LOG_DIR" >&2
  exit 1
fi
date -Is > "$ROOT/position_ablation_queue/COMPLETE"
