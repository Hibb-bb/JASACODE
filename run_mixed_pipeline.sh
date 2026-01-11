#!/bin/bash
#SBATCH --account=p32626
#SBATCH --job-name=c100_l12_mixed
#SBATCH --nodes=1
#SBATCH --output=/projects/p32626/JASACODE/slurm_log/slurm_mixed_eval_%j.out
#SBATCH --error=/projects/p32626/JASACODE/slurm_log/slurm_mixed_eval_%j.err
#SBATCH --time=01:00:00      
#SBATCH --partition=gengpu   
#SBATCH --gres=gpu:h100:1      
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=10G
#SBATCH --mail-type=ALL ## you can receive e-mail alerts from SLURM when your job begins and when your job finishes (completed, failed, etc)
#SBATCH --mail-user=b0976960890@gmail.com ## your email

set -e  # Exit on error

echo "=============================================="
echo "Mixed Structure Training & Evaluation Pipeline"
echo "=============================================="
echo ""

# Configuration
SEED="${1:-1111}"
CONTEXT_LEN="${2:-100}"
TRAIN_SIZE="${3:-20000}"
TRAIN_STEPS="${4:-50000}"

echo "Configuration:"
echo "  Seed: $SEED"
echo "  Context length: $CONTEXT_LEN"
echo "  Train size: $TRAIN_SIZE"
echo "  Train steps: $TRAIN_STEPS"
echo ""

# Output directories
RUN_NAME="mixed_seed${SEED}_ctx${CONTEXT_LEN}_train${TRAIN_SIZE}"
TRAIN_DIR="runs/mixed/${RUN_NAME}"
EVAL_DIR="eval_results/${RUN_NAME}"

echo "Output directories:"
echo "  Training: $TRAIN_DIR"
echo "  Evaluation: $EVAL_DIR"
echo ""

# # Step 1: Train
# echo "=============================================="
# echo "Step 1/3: Training"
# echo "=============================================="

# python train_mixed.py \
#   --batch-size 64 \
#   --context-len $CONTEXT_LEN \
#   --train-step $TRAIN_STEPS \
#   --init-lr 1e-4 \
#   --train-size $TRAIN_SIZE \
#   --test-size 5000 \
#   --output-dir "runs/mixed" \
#   --seed $SEED

# echo ""
# echo "✓ Training complete"
# echo ""

# # Find checkpoint
# CHECKPOINT=$(find "$TRAIN_DIR" -name "best.ckpt" -type f | head -n 1)

# if [ -z "$CHECKPOINT" ]; then
#   echo "Error: Could not find checkpoint in $TRAIN_DIR"
#   exit 1
# fi

# echo "Found checkpoint: $CHECKPOINT"
# echo ""

# # Step 2: Evaluate
# echo "=============================================="
# echo "Step 2/3: Evaluation"
# echo "=============================================="

# python eval_mixed.py \
#   --checkpoint "$CHECKPOINT" \
#   --output-dir "$EVAL_DIR" \
#   --test-size 5000 \
#   --seed 123 \
#   --context-lens 1 2 5 10 20 50 100 200 300 400 500 \
#   --batch-size 512

# echo ""
# echo "✓ Evaluation complete"
# echo ""

# Step 3: Analyze
echo "=============================================="
echo "Step 3/3: Analysis"
echo "=============================================="

python analyze_mixed_eval.py --eval-dir "$EVAL_DIR"

echo ""
echo "✓ Analysis complete"
echo ""

# Summary
echo "=============================================="
echo "PIPELINE COMPLETE"
echo "=============================================="
echo ""
echo "Results:"
echo "  Training logs: $TRAIN_DIR"
echo "  Checkpoint: $CHECKPOINT"
echo "  Evaluation CSVs: $EVAL_DIR"
echo "  Plots: $EVAL_DIR/*.png"
echo "  Statistics: $EVAL_DIR/summary_statistics.txt"
echo ""
echo "To view results:"
echo "  cat $EVAL_DIR/summary_statistics.txt"
echo "  ls $EVAL_DIR/*.png"
