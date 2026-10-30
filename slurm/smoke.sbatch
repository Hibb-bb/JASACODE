#!/bin/bash
#SBATCH --account=p32593
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --job-name=jasa-smoke
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err
set -uo pipefail
cd /gpfs/projects/p32593/JASACODE
source .venv/bin/activate
nvidia-smi --query-gpu=name --format=csv,noheader
OUT=runs_smoke
rm -rf $OUT
run() { echo "=================== $*"; /usr/bin/time -f "elapsed %es maxrss %MKB" "$@" 2>&1 | grep -v -i "warning" | tail -n 25; }
# A1: gpt2 L2H1 D256, TV loss
run python train.py --graph general --context-len 200 --batch-size 64 --train-step 300 --warmup-steps 50 --train-size 2000 --test-size 64 --output-dir $OUT --seed 1 --num-layers 2 --arch gpt2 --n-head 1
# A2: theory model, TV loss
run python train.py --graph general --context-len 200 --batch-size 64 --train-step 300 --warmup-steps 50 --train-size 2000 --test-size 64 --output-dir $OUT --seed 1 --num-layers 2 --arch theory
# B: gpt2 L4H8 BCE, ctx 500, bs 128
run python train_pred.py --graph general --context-len 500 --batch-size 128 --train-step 300 --warmup-steps 50 --train-size 1000 --test-size 64 --output-dir $OUT/pred --seeds 1 --no-plot --num-layers 4 --cpt-prior uniform
# C: theory BCE
run python train_pred.py --graph general --context-len 500 --batch-size 128 --train-step 300 --warmup-steps 50 --train-size 1000 --test-size 64 --output-dir $OUT/pred --seeds 1 --no-plot --num-layers 2 --arch theory --cpt-prior uniform
# timing probe: 1000 steps of the heaviest config (B) and of A1 at ctx 200
run python train_pred.py --graph tree --context-len 500 --batch-size 128 --train-step 1000 --warmup-steps 50 --train-size 1000 --test-size 16 --output-dir $OUT/pred_t --seeds 1 --no-plot --num-layers 4 --cpt-prior uniform
run python train.py --graph tree --context-len 200 --batch-size 64 --train-step 1000 --warmup-steps 50 --train-size 2000 --test-size 16 --output-dir $OUT/t --seed 1 --num-layers 4
find $OUT -name "*_eval_tv.csv" | sed 's/^/CSV: /'
