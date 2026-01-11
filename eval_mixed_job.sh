#!/bin/bash
#SBATCH --account=p32626
#SBATCH --job-name=jasa_mixed
#SBATCH --nodes=1
#SBATCH --output=/projects/p32626/JASACODE/slurm_log/slurm_mixed_eval_%j.out
#SBATCH --error=/projects/p32626/JASACODE/slurm_log/slurm_mixed_eval_%j.err
#SBATCH --time=01:00:00      
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
echo "Mixed Structure Model Evaluation"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Date: $(date)"
echo "=========================================="

# Configuration
CHECKPOINT_PATH="$1"  # Pass checkpoint path as first argument
OUTPUT_DIR="${2:-eval_results_mixed}"  # Default output dir

if [ -z "$CHECKPOINT_PATH" ]; then
  echo "Error: No checkpoint path provided"
  echo "Usage: sbatch eval_mixed_job.sh <checkpoint_path> [output_dir]"
  exit 1
fi

if [ ! -f "$CHECKPOINT_PATH" ]; then
  echo "Error: Checkpoint file not found: $CHECKPOINT_PATH"
  exit 1
fi

echo ""
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Run evaluation
echo "Starting evaluation on all structures..."
python eval_mixed.py \
  --checkpoint "$CHECKPOINT_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --test-size 5000 \
  --seed 123 \
  --context-lens 1 2 5 10 20 50 100 200 300 400 500 \
  --batch-size 512

echo ""
echo "=========================================="
echo "Evaluation completed!"
echo "Date: $(date)"
echo "=========================================="
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "To analyze results, run:"
echo "  python analyze_mixed_eval.py --eval-dir $OUTPUT_DIR"
