#!/bin/bash
#SBATCH --job-name=jasacode_train
#SBATCH --partition=djob_a100
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=40G
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

# Activate virtual environment
source .venv/bin/activate

# Run training
CUDA_LAUNCH_BLOCKING=1 python train.py \
  --batch-size 64 \
  --context-len 100 \
  --graph tree \
  --train-step 50000 \
  --init-lr 1e-4 \
  --train-size 10000 \
  --test-size 1000 \
  --output-dir runs/ \
  --seed 1111

CUDA_LAUNCH_BLOCKING=1 python train.py \
  --batch-size 64 \
  --context-len 100 \
  --graph chain \
  --train-step 50000 \
  --init-lr 1e-4 \
  --train-size 10000 \
  --test-size 1000 \
  --output-dir runs/ \
  --seed 1111


  CUDA_LAUNCH_BLOCKING=1 python train.py \
  --batch-size 64 \
  --context-len 100 \
  --graph general \
  --train-step 50000 \
  --init-lr 1e-4 \
  --train-size 10000 \
  --test-size 1000 \
  --output-dir runs/ \
  --seed 1111

