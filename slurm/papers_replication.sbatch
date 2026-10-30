#!/bin/bash
#SBATCH --account=p32593
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --job-name=papers
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err
# Usage: sbatch --job-name=papers-clean slurm/papers_replication.sbatch --out-dir runs_papers/clean
#        sbatch --job-name=papers-real  slurm/papers_replication.sbatch --out-dir runs_papers/realistic --realistic
set -uo pipefail
cd /gpfs/projects/p32593/JASACODE
source .venv/bin/activate
echo "[$(date)] host=$(hostname) args=$*"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python -u papers_replication.py "$@"
echo "[$(date)] exit $?"
