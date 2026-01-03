#!/bin/bash
#SBATCH --job-name=jasacode_train
#SBATCH --partition=job_a100
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=40G
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

# Activate virtual environment
source .venv/bin/activate



for seed in 1111 2222 3333 4444 5555; do

  CUDA_LAUNCH_BLOCKING=1 python train.py \
  --batch-size 64 \
  --context-len 10 \
  --graph general \
  --train-step 50000 \
  --init-lr 1e-4 \
  --train-size 20000 \
  --test-size 5000 \
  --output-dir runs/ \
  --seed $seed

  CUDA_LAUNCH_BLOCKING=1 python train.py \
    --batch-size 64 \
    --context-len 10 \
    --graph tree \
    --train-step 50000 \
    --init-lr 1e-4 \
    --train-size 20000 \
    --test-size 5000 \
    --output-dir runs/ \
    --seed $seed


  CUDA_LAUNCH_BLOCKING=1 python train.py \
    --batch-size 64 \
    --context-len 10 \
    --graph tree \
    --train-step 50000 \
    --init-lr 1e-4 \
    --train-size 20000 \
    --test-size 5000 \
    --output-dir runs/ \
    --seed $seed






for seed in 1111 2222 3333 4444 5555; do

  CUDA_LAUNCH_BLOCKING=1 python train.py \
  --batch-size 64 \
  --context-len 20 \
  --graph general \
  --train-step 50000 \
  --init-lr 1e-4 \
  --train-size 20000 \
  --test-size 5000 \
  --output-dir runs/ \
  --seed $seed

  CUDA_LAUNCH_BLOCKING=1 python train.py \
    --batch-size 64 \
    --context-len 20 \
    --graph tree \
    --train-step 50000 \
    --init-lr 1e-4 \
    --train-size 20000 \
    --test-size 5000 \
    --output-dir runs/ \
    --seed $seed


  CUDA_LAUNCH_BLOCKING=1 python train.py \
    --batch-size 64 \
    --context-len 20 \
    --graph tree \
    --train-step 50000 \
    --init-lr 1e-4 \
    --train-size 20000 \
    --test-size 5000 \
    --output-dir runs/ \
    --seed $seed