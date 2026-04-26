#!/bin/bash
#SBATCH --account=p32593
#SBATCH --job-name=pred_jasa
#SBATCH --nodes=1
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=24:00:00
#SBATCH --output=pred.out
#SBATCH --error=pred.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hibb@u.northwestern.edu


source .venv/bin/activate

python train_pred.py --graph tree --context-len 500 --num-layers 4 --train-size 1000

python train_pred.py --graph chain --context-len 500 --num-layers 4 --train-size 1000

python train_pred.py --graph general --context-len 500 --num-layers 4 --train-size 1000
