#!/bin/bash
#SBATCH --account=p32593
#SBATCH --job-name=sachs_eval
#SBATCH --nodes=1
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=5:00:00
#SBATCH --output=eval.out
#SBATCH --error=eval.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hibb@u.northwestern.edu


module purge

cd /projects/b1094/ywl7940/JASACODE

source ./.venv/bin/activate

# uv run python train_sachs.py --train-step 100000 --batch-size 64 --context-len 1000 --seed 1111 --num-layers 4 --train-size 1

# uv run python3 eval_sachs_real.py \
# --ckpt-path "/projects/b1094/ywl7940/JASACODE/outputs/sachs/seed_1111/500/L12/1/logs/version_0/checkpoints/best.ckpt" \
# --disc-data-dir Sachs/disc_data \
# --output-dir runs/sachs_real_eval_seed1111 \
# --seed 1111



TRAIN_SIZE=100


python3 eval_sachs_real.py \
      --ckpt-path "/projects/b1094/ywl7940/JASACODE/outputs/sachs/seed_1111/1000/L2/100/logs/version_1/checkpoints/best.ckpt" \
      --disc-data-dir Sachs/disc_data \
      --output-dir runs/sachs_real_eval_seed1111 \
      --seed 1111

for seed in 2222 3333 4444 5555; do

    uv run python3 eval_sachs_real.py \
      --ckpt-path "/projects/b1094/ywl7940/JASACODE/outputs/sachs/seed_${seed}/1000/L2/100/logs/version_0/checkpoints/best.ckpt" \
      --disc-data-dir Sachs/disc_data \
      --output-dir runs/sachs_real_eval_seed$seed \
      --seed $seed

done