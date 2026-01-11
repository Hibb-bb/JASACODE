#!/bin/bash
#SBATCH --job-name=jasacode_compare
#SBATCH --partition=job_a100
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --output=slurm_compare_%j.out
#SBATCH --error=slurm_compare_%j.err

# Activate virtual environment
source .venv/bin/activate

echo "=========================================="
echo "Single vs Mixed Graph Structure Comparison"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Date: $(date)"
echo "=========================================="

# Configuration
CONTEXT_LEN=10
SEEDS=(1111 2222 3333)

for seed in "${SEEDS[@]}"; do
  
  echo ""
  echo "=========================================="
  echo "Experiment with seed=$seed"
  echo "=========================================="
  
  # 1. Train on single structures (tree, chain, general)
  for graph_type in tree chain general; do
    echo ""
    echo ">> Training SINGLE structure: $graph_type (seed=$seed)"
    
    CUDA_LAUNCH_BLOCKING=1 python train.py \
      --batch-size 64 \
      --context-len $CONTEXT_LEN \
      --graph $graph_type \
      --train-step 50000 \
      --init-lr 1e-4 \
      --train-size 20000 \
      --test-size 5000 \
      --output-dir runs/single/ \
      --seed $seed \
      --log-interval 1000
    
    echo "   Completed: $graph_type"
  done
  
  # 2. Train on mixed structures
  echo ""
  echo ">> Training MIXED structures (seed=$seed)"
  
  CUDA_LAUNCH_BLOCKING=1 python train_mixed.py \
    --batch-size 64 \
    --context-len $CONTEXT_LEN \
    --train-step 50000 \
    --init-lr 1e-4 \
    --train-size 20000 \
    --test-size 5000 \
    --output-dir runs/mixed/ \
    --seed $seed \
    --log-interval 1000
  
  echo "   Completed: mixed structures"
  
done

echo ""
echo "=========================================="
echo "All comparison experiments completed!"
echo "Date: $(date)"
echo "=========================================="
echo ""
echo "Results saved to:"
echo "  - Single structure: runs/single/"
echo "  - Mixed structures: runs/mixed/"
echo ""
