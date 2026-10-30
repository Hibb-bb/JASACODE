#!/bin/bash
#SBATCH --account=p32593
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --job-name=jasa-rebuttal
#SBATCH --output=slurm/logs/%x-%A_%a.out
#SBATCH --error=slurm/logs/%x-%A_%a.err
# Submit with:  sbatch --array=1-N slurm/array.sbatch slurm/configs/<file>.txt
bash slurm/run_one.sh "$1" "$SLURM_ARRAY_TASK_ID"
