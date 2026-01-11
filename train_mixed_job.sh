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
#SBATCH --mail-type=ALL ## you can receive e-mail alerts from SLURM when your job begins and when your job finishes (completed, failed, etc)
#SBATCH --mail-user=b0976960890@gmail.com ## your email


# Activate virtual environment
source .venv/bin/activate

echo "=========================================="
echo "Mixed Graph Structure Training"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Date: $(date)"
echo "=========================================="

# Run mixed graph training with different context lengths and seeds
for context_len in 10 20; do
  for seed in 1111 2222 3333 4444 5555; do
    
    echo ""
    echo "----------------------------------------"
    echo "Training: context_len=$context_len, seed=$seed"
    echo "----------------------------------------"
    
    CUDA_LAUNCH_BLOCKING=1 python train_mixed.py \
      --batch-size 64 \
      --context-len $context_len \
      --train-step 50000 \
      --init-lr 1e-4 \
      --train-size 20000 \
      --test-size 5000 \
      --output-dir runs/mixed/ \
      --seed $seed
    
    echo "Completed: context_len=$context_len, seed=$seed"
    
  done
done

echo ""
echo "=========================================="
echo "All mixed graph training completed!"
echo "Date: $(date)"
echo "=========================================="
