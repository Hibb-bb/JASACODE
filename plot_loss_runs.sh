#!/bin/bash
set -euo pipefail

# Plot training loss + TV for tree, chain, and general (seed 1111 only).
#
# Matches train_pred.py layout (see pred.sh — same flags unless you override):
#   outputs_pred/<graph>/seed_<seed>/<context>/L<layers>/<train_size>/beta<alpha>/pred_bce/logs/
#
# pred.sh uses:
#   --context-len 500 --num-layers 4 --train-size 1000
#   (and default --cpt-beta-alpha 0.4 in train_pred.py if you do not pass it)
#
# Context rule for non-pred runs:
#   outputs_pred -> 500
#   runs / outputs -> 200
#
# Usage:
#   ./plot_loss_runs.sh [BASE_DIR [NUM_LAYERS [TRAIN_SIZE [CPT_BETA_ALPHA]]]]
#
# Examples:
#   ./plot_loss_runs.sh
#   ./plot_loss_runs.sh outputs_pred 4 1000 0.4
#   ./plot_loss_runs.sh outputs_pred 4 500 0.4   # if you only trained with --train-size 500

ROOT="/projects/b1094/ywl7940/JASACODE"
BASE_DIR="${1:-outputs_pred}"
NUM_LAYERS="${2:-4}"
TRAIN_SIZE="${3:-1000}"
CPT_BETA_ALPHA="${4:-0.4}"

GRAPHS=(tree chain general)
SEEDS=(1111)

if [[ "$BASE_DIR" == "outputs_pred" ]]; then
  CONTEXT="500"
  LOSS_KIND="crossentropy"
else
  CONTEXT="200"
  LOSS_KIND="auto"
fi

OUT_DIR="${ROOT}/imgs/loss_curves"
mkdir -p "${OUT_DIR}"

logs_path_for() {
  local graph="$1" seed="$2"
  if [[ "$BASE_DIR" == "outputs_pred" ]]; then
    echo "${ROOT}/${BASE_DIR}/${graph}/seed_${seed}/${CONTEXT}/L${NUM_LAYERS}/${TRAIN_SIZE}/beta${CPT_BETA_ALPHA}/pred_bce/logs"
  else
    echo "${ROOT}/${BASE_DIR}/${graph}/seed_${seed}/${CONTEXT}/L${NUM_LAYERS}/${TRAIN_SIZE}/logs"
  fi
}

plotted=0
for GRAPH in "${GRAPHS[@]}"; do
  paths=()
  labels=()
  for seed in "${SEEDS[@]}"; do
    p="$(logs_path_for "$GRAPH" "$seed")"
    if [[ -d "$p" ]]; then
      paths+=("$p")
      labels+=("seed_${seed}")
    else
      echo "WARN: missing ${p} (graph=${GRAPH}) — skip this graph or fix TRAIN_SIZE/BETA/LAYERS/CONTEXT."
    fi
  done

  if [[ "${#paths[@]}" -eq 0 ]]; then
    echo "WARN: no logs for graph=${GRAPH} under ${BASE_DIR}; skipping plot."
    continue
  fi

  out_png="${OUT_DIR}/${GRAPH}_${BASE_DIR}_ctx${CONTEXT}_L${NUM_LAYERS}_tr${TRAIN_SIZE}_beta${CPT_BETA_ALPHA}.png"
  python3 "${ROOT}/plot_loss.py" \
    --run-dir "${paths[@]}" \
    --label "${labels[@]}" \
    --loss-kind "${LOSS_KIND}" \
    --output "${out_png}"

  echo "Wrote: ${out_png}"
  plotted=$((plotted + 1))
done

if [[ "$plotted" -eq 0 ]]; then
  echo "ERROR: no figures written; check paths vs pred.sh (train-size, layers, beta)." >&2
  exit 1
fi
