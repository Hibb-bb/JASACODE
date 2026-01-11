#!/bin/bash
#SBATCH --job-name=jasacode_pilot
#SBATCH --partition=job_a100
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --output=slurm_pilot_%j.out
#SBATCH --error=slurm_pilot_%j.err

# Activate virtual environment
source .venv/bin/activate

echo "=========================================="
echo "Pilot Test: Mixed Graph Structure Training"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Date: $(date)"
echo "=========================================="

# Quick pilot test with single configuration
SEED=1111
CONTEXT_LEN=10

echo ""
echo "Running pilot test with:"
echo "  - Seed: $SEED"
echo "  - Context length: $CONTEXT_LEN"
echo "  - Training steps: 10000 (reduced for testing)"
echo ""

CUDA_LAUNCH_BLOCKING=1 python train_mixed.py \
  --batch-size 64 \
  --context-len $CONTEXT_LEN \
  --train-step 10000 \
  --init-lr 1e-4 \
  --train-size 5000 \
  --test-size 1000 \
  --output-dir runs/pilot/ \
  --seed $SEED \
  --log-interval 500

echo ""
echo "=========================================="
echo "Pilot test completed!"
echo "Date: $(date)"
echo "=========================================="
echo ""
echo "Check results at: runs/pilot/mixed_ctx${CONTEXT_LEN}_seed${SEED}/"
echo ""
