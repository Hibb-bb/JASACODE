#!/bin/bash
#SBATCH --account=p32593
#SBATCH --job-name=10-node
#SBATCH --nodes=1
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=20:00:00
#SBATCH --output=10-node.out
#SBATCH --error=10-node.err

source .venv/bin/activate

uv run python train_pred.py --graph tree --context-len 200 --num-layers 4 --train-size 1000
uv run python train_pred.py --graph chain --context-len 200 --num-layers 4 --train-size 1000
uv run python train_pred.py --graph general --context-len 200 --num-layers 4 --train-size 1000
