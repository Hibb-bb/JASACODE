#!/bin/bash
# Usage: run_one.sh <config-file> <line-number>
# Each line of the config file is a full command (relative to repo root).
set -euo pipefail
CFG=$1; IDX=$2
cd /gpfs/projects/p32593/JASACODE
source .venv/bin/activate
CMD=$(sed -n "${IDX}p" "$CFG")
echo "[$(date)] host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-none}"
echo "[$(date)] $CMD"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
eval "$CMD"
echo "[$(date)] done"
