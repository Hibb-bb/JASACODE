#!/bin/bash
#SBATCH --account=p32626
#SBATCH --job-name=jasa_mixed
#SBATCH --nodes=1
#SBATCH --output=/projects/p32626/JASACODE/slurm_log/slurm_mixed_%j.out
#SBATCH --error=/projects/p32626/JASACODE/slurm_log/slurm_mixed_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=b0976960890@gmail.com

# Activate virtual environment
source .venv/bin/activate

echo "=========================================="
echo "Mixed Graph Structure Training"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Date: $(date)"
echo "=========================================="

# Run mixed graph training with dynamic context length and multiple seeds
# Parameters match train_job.sh for consistency
for seed in 1111 2222 3333; do
  
  echo ""
  echo "----------------------------------------"
  echo "Training mixed structures: seed=$seed"
  echo "----------------------------------------"
  
  CUDA_LAUNCH_BLOCKING=1 python train_mixed.py \
    --batch-size 64 \
    --min-context-len 5 \
    --max-context-len 500 \
    --train-step 15000 \
    --init-lr 3e-4 \
    --train-size 20000 \
    --test-size 5000 \
    --output-dir runs/best/step_15k_tvloss \
    --warmup-steps 2000 \
    --min-lr 1e-6 \
    --seed $seed
  
  echo "Completed: seed=$seed"
  
done

echo ""
echo "=========================================="
echo "All mixed graph training completed!"
echo "Date: $(date)"
echo "=========================================="
