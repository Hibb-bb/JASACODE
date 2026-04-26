#!/bin/bash
#SBATCH --account=p32593
#SBATCH --job-name=sachs_real_eval_5seeds
#SBATCH --nodes=1
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=8:00:00
#SBATCH --output=sachs_real_eval_5seeds.out
#SBATCH --error=sachs_real_eval_5seeds.err

set -euo pipefail

cd /projects/b1094/ywl7940/JASACODE

source ./.venv/bin/activate

# ---- Config: match your training output folder layout
GRAPH_DIR="sachs"
CONTEXT="1000"      # e.g. 500 or 1000
NUM_LAYERS="2"      # e.g. 2 or 12
TRAIN_SIZE="100"    # e.g. 1 or 100

DISC_DIR="Sachs/disc_data"
OUT_BASE="runs/sachs_real_eval"

SEEDS=(1111 2222 3333 4444 5555)

latest_ckpt () {
  # Args: seed
  local seed="$1"
  local base="outputs/${GRAPH_DIR}/seed_${seed}/${CONTEXT}/L${NUM_LAYERS}/${TRAIN_SIZE}/logs"
  if [[ ! -d "$base" ]]; then
    echo "ERROR: logs dir not found: $base" >&2
    return 2
  fi
  local latest_version
  latest_version="$(ls -d "${base}/version_"* 2>/dev/null | sort -V | tail -n 1)"
  if [[ -z "${latest_version}" ]]; then
    echo "ERROR: no version_* dirs under: $base" >&2
    return 3
  fi
  local ckpt="${latest_version}/checkpoints/best.ckpt"
  if [[ ! -f "$ckpt" ]]; then
    echo "ERROR: checkpoint not found: $ckpt" >&2
    return 4
  fi
  echo "$ckpt"
}

for seed in "${SEEDS[@]}"; do
  ckpt_path="$(latest_ckpt "$seed")"
  out_dir="${OUT_BASE}/seed_${seed}"
  mkdir -p "$out_dir"
  echo "Seed ${seed}: ${ckpt_path}"
  python3 eval_sachs_real.py \
    --ckpt-path "$ckpt_path" \
    --disc-data-dir "$DISC_DIR" \
    --output-dir "$out_dir" \
    --seed "$seed"
done

# Aggregate plots across seeds (quick_plot-style) for each treatment
python3 quick_plot_sachs.py \
  --base-dir "$OUT_BASE" \
  --out-dir "imgs/sachs_real_avg5" \
  --seeds "${SEEDS[@]}"

