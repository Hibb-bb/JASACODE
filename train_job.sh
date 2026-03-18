#!/bin/bash
#SBATCH --job-name=sachs_train
#SBATCH --partition=job_a100
#SBATCH --time=5:00:00
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=40G
#SBATCH --output=run.out
#SBATCH --error=run.err

source .venv/bin/activate

for seed in 1111 2222 3333 4444 5555; do
    uv run python train_sachs.py --train-step 200000 --batch-size 16 --context-len 200 --seed $seed
done

# CUDA_LAUNCH_BLOCKING=1 python train.py \
#   --batch-size 64 \
#   --min-context-len 50 \
#   --max-context-len 500 \
#   --graph tree \
#   --train-step 100 \
#   --init-lr 1e-4 \
#   --train-size 20000 \
#   --test-size 5000 \
#   --output-dir runs/ \
#   --warmup-steps 2000 \
#   --min-lr 1e-6 \
#   --init-lr 3e-4 \
#   --seed 3333


# for seed in 1234; do

#   CUDA_LAUNCH_BLOCKING=1 python train.py \
#     --batch-size 16 \
#     --context-len 200 \
#     --graph tree \
#     --train-step 10000 \
#     --init-lr 1e-4 \
#     --train-size 20000 \
#     --test-size 5000 \
#     --output-dir runs/ \
#     --warmup-steps 2000 \
#     --min-lr 1e-6 \
#     --init-lr 3e-4 \
#     --seed 1234


#   CUDA_LAUNCH_BLOCKING=1 python train.py \
#     --batch-size 16 \
#     --context-len 200 \
#     --graph tree5 \
#     --train-step 10000 \
#     --init-lr 1e-4 \
#     --train-size 20000 \
#     --test-size 5000 \
#     --output-dir runs/ \
#     --warmup-steps 2000 \
#     --min-lr 1e-6 \
#     --init-lr 3e-4 \
#     --seed 1234

# done