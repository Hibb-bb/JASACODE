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



for seed in 1111 2222 3333 4444 5555; do

  CUDA_LAUNCH_BLOCKING=1 python train.py \
    --batch-size 64 \
    --min-context-len 5 \
    --max-context-len 500 \
    --graph tree \
    --train-step 10000 \
    --init-lr 1e-4 \
    --train-size 20000 \
    --test-size 5000 \
    --output-dir runs/ \
    --warmup-steps 2000 \
    --min-lr 1e-6 \
    --init-lr 3e-4 \
    --seed $seed


  CUDA_LAUNCH_BLOCKING=1 python train.py \
    --batch-size 64 \
    --min-context-len 5 \
    --max-context-len 500 \
    --graph chain \
    --train-step 10000 \
    --init-lr 1e-4 \
    --train-size 20000 \
    --test-size 5000 \
    --output-dir runs/ \
    --warmup-steps 2000 \
    --min-lr 1e-6 \
    --init-lr 3e-4 \
    --seed $seed


  CUDA_LAUNCH_BLOCKING=1 python train.py \
    --batch-size 64 \
    --min-context-len 5 \
    --max-context-len 500 \
    --graph general \
    --train-step 10000 \
    --init-lr 1e-4 \
    --train-size 20000 \
    --test-size 5000 \
    --output-dir runs/ \
    --warmup-steps 2000 \
    --min-lr 1e-6 \
    --init-lr 3e-4 \
    --seed $seed



done




for seed in 1111 2222 3333 4444 5555; do

  CUDA_LAUNCH_BLOCKING=1 python train.py \
    --batch-size 64 \
    --min-context-len 5 \
    --max-context-len 500 \
    --graph tree \
    --train-step 10000 \
    --init-lr 1e-4 \
    --train-size 20000 \
    --test-size 5000 \
    --output-dir runs/ \
    --warmup-steps 2000 \
    --min-lr 1e-6 \
    --init-lr 3e-4 \
    --seed $seed


  CUDA_LAUNCH_BLOCKING=1 python train.py \
    --batch-size 64 \
    --min-context-len 5 \
    --max-context-len 500 \
    --graph chain \
    --train-step 10000 \
    --init-lr 1e-4 \
    --train-size 20000 \
    --test-size 5000 \
    --output-dir runs/ \
    --warmup-steps 2000 \
    --min-lr 1e-6 \
    --init-lr 3e-4 \
    --seed $seed


  CUDA_LAUNCH_BLOCKING=1 python train.py \
    --batch-size 64 \
    --min-context-len 5 \
    --max-context-len 500 \
    --graph general \
    --train-step 10000 \
    --init-lr 1e-4 \
    --train-size 20000 \
    --test-size 5000 \
    --output-dir runs/ \
    --warmup-steps 2000 \
    --min-lr 1e-6 \
    --init-lr 3e-4 \
    --seed $seed


done
