#!/bin/bash
#SBATCH --account=p32593
#SBATCH --job-name=jasa_code_chain5
#SBATCH --nodes=1
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=10:00:00
#SBATCH --output=run.out
#SBATCH --error=run.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hibb@u.northwestern.edu


module purge

cd /projects/b1094/ywl7940/JASACODE

source ./.venv/bin/activate

# uv run python train_sachs.py --train-step 100000 --batch-size 16 --context-len 200 --seed 1111



# uv run python3 eval_sachs_real.py \
#   --ckpt-path /projects/b1094/ywl7940/JASACODE/outputs/sachs/seed_1111/200/L12/1000/logs/version_3/checkpoints/best.ckpt \
#   --disc-data-dir Sachs/disc_data \
#   --output-dir runs/sachs_real_eval_seed1111 \
#   --seed 1111

# for seed in 1111 2222 3333 4444 5555; do

#     uv run python train_sachs.py --train-step 200000 --batch-size 16 --context-len 200 --seed $seed

#     uv run python3 eval_sachs_real.py \
#       --ckpt-path /projects/b1094/ywl7940/JASACODE/outputs/sachs/seed_$seed/200/L12/1000/logs/version_4/checkpoints/best.ckpt \
#       --disc-data-dir Sachs/disc_data \
#       --output-dir runs/sachs_real_eval_seed$seed \
#       --seed $seed

# done

for seed in 1111 2222 3333 4444 5555; do

  uv run python train_pred.py --graph tree --context-len 200 --train-step 100000 --train-size 1 --seed 0

done
