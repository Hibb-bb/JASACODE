#!/bin/bash
#SBATCH --job-name=jasacode_train
#SBATCH --partition=job_a100
#SBATCH --time=18:00:00
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=40G
#SBATCH --output=run.out
#SBATCH --error=run.err

# Activate virtual environment
source .venv/bin/activate



for seed in 1111; do

  CUDA_LAUNCH_BLOCKING=1 python train.py \
  --batch-size 64 \
  --context-len 50 \
  --graph general \
  --train-step 30000 \
  --init-lr 1e-4 \
  --train-size 20000 \
  --test-size 5000 \
  --output-dir runs/ \
  --seed $seed

  CUDA_LAUNCH_BLOCKING=1 python train.py \
    --batch-size 64 \
    --context-len 50 \
    --graph tree \
    --train-step 30000 \
    --init-lr 1e-4 \
    --train-size 20000 \
    --test-size 5000 \
    --output-dir runs/ \
    --seed $seed


  CUDA_LAUNCH_BLOCKING=1 python train.py \
    --batch-size 64 \
    --context-len 50 \
    --graph chain \
    --train-step 30000 \
    --init-lr 1e-4 \
    --train-size 20000 \
    --test-size 5000 \
    --output-dir runs/ \
    --seed $seed



done


for seed in 1111; do

  CUDA_LAUNCH_BLOCKING=1 python train.py \
  --batch-size 64 \
  --context-len 100 \
  --graph general \
  --train-step 30000 \
  --init-lr 1e-4 \
  --train-size 20000 \
  --test-size 5000 \
  --output-dir runs/ \
  --seed $seed

  CUDA_LAUNCH_BLOCKING=1 python train.py \
    --batch-size 64 \
    --context-len 100 \
    --graph tree \
    --train-step 30000 \
    --init-lr 1e-4 \
    --train-size 20000 \
    --test-size 5000 \
    --output-dir runs/ \
    --seed $seed


  CUDA_LAUNCH_BLOCKING=1 python train.py \
    --batch-size 64 \
    --context-len 100 \
    --graph chain \
    --train-step 30000 \
    --init-lr 1e-4 \
    --train-size 20000 \
    --test-size 5000 \
    --output-dir runs/ \
    --seed $seed

done