#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp
SHAPE_SCRIPT="$ROOT/train_shape_shared_trunk.py"
MAG_SCRIPT="$ROOT/train_dense_motif_magnitude.py"
SHAPE_OUT="$ROOT/shape_motif_nopos_equalce5"
MAG_OUT="$ROOT/dense_motif_nopos5"
LOG_DIR="$ROOT/position_ablation_queue/logs"
mkdir -p "$LOG_DIR"

wait_for_pid() {
  local pid="$1"
  while kill -0 "$pid" 2>/dev/null; do
    sleep 30
  done
}

run_fold() {
  local gpu="$1"
  local fold="$2"
  CUDA_VISIBLE_DEVICES="$gpu" python "$SHAPE_SCRIPT" \
    --modality motif_nopos --fold "$fold" --objective equal_ce \
    --epochs 30 --patience 5 --batch-size 1024 --accum-steps 4 \
    --lr 0.001 --p-threshold 0.001 --rc-prob 0.0 \
    --output-root "$SHAPE_OUT" \
    >"$LOG_DIR/shape_nopos_fold$((fold + 1)).log" 2>&1

  CUDA_VISIBLE_DEVICES="$gpu" python "$MAG_SCRIPT" \
    --variant conv_nopos --fold "$fold" --epochs 25 --patience 5 \
    --batch-size 1024 --accum-steps 2 --lr 0.0003 --seed-offset 100 \
    --rc-prob 0.0 --output-root "$MAG_OUT" \
    >"$LOG_DIR/magnitude_nopos_fold$((fold + 1)).log" 2>&1
}

worker0() {
  wait_for_pid 19689
  run_fold 0 0
  run_fold 0 4
}

worker1() {
  wait_for_pid 19524
  run_fold 1 1
}

worker2() {
  wait_for_pid 18730
  run_fold 2 2
}

worker3() {
  wait_for_pid 18731
  run_fold 3 3
}

worker0 &
worker1 &
worker2 &
worker3 &
wait
date -Is > "$ROOT/position_ablation_queue/COMPLETE"
