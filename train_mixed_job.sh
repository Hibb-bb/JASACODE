#!/bin/bash
#SBATCH --account=p32626
#SBATCH --job-name=7_nodes_new_structure
#SBATCH --nodes=1
#SBATCH --output=/projects/p32626/JASACODE/slurm_log/slurm_mixed_%j.out
#SBATCH --error=/projects/p32626/JASACODE/slurm_log/slurm_mixed_%j.err
#SBATCH --time=03:00:00
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=b0976960890@gmail.com

# Activate virtual environment

cd /projects/p32626/JASACODE
source .venv/bin/activate

# cd /projects/p32626/JASACODE

echo "=========================================="
echo "Mixed Graph Structure Training"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Date: $(date)"
echo "=========================================="

# Run mixed graph training with dynamic context length and multiple seeds
# Parameters match train_job.sh for consistency
for seed in 1111 2222 3333 4444 5555; do

  echo ""
  echo "----------------------------------------"
  echo "Training mixed structures: seed=$seed"
  echo "----------------------------------------"
  
  CUDA_LAUNCH_BLOCKING=1 python train_mixed.py \
    --batch-size 16 \
    --context-len 500 \
    --train-step 50000 \
    --init-lr 3e-4 \
    --train-size 20000 \
    --test-size 5000 \
    --output-dir runs/best/7_nodes_step_50k_tvloss_new_structure_ctx500 \
    --warmup-steps 2000 \
    --min-lr 1e-6 \
    --num-nodes 7 \
    --target-index 6 \
    --seed $seed
  
  echo "Completed training: seed=$seed"
  
  # Generate training loss plot
  echo "Generating training loss plot for seed=$seed"
  python plot_training_loss.py \
    --metrics_csv runs/best/7_nodes_step_50k_tvloss_new_structure_ctx500/mixed_seed${seed}_ctx500_train20000/logs/version_0/metrics.csv \
    -o runs/best/7_nodes_step_50k_tvloss_new_structure_ctx500/mixed_seed${seed}_ctx500_train20000/training_loss_plot.png
  
  echo "Completed all tasks: seed=$seed"
  
done

echo ""
echo "=========================================="
echo "All seeds done — generating aggregate plots (mean ± std across seeds)"
echo "=========================================="

OUTPUT_DIR=runs/best/7_nodes_step_50k_tvloss_new_structure_ctx500

for structure in tree chain general; do
  echo "Plotting $structure ..."
  python plot_eval_mixed.py \
    --output-dir $OUTPUT_DIR \
    --structure $structure \
    --context 500 \
    --train-size 20000
done

echo ""
echo "=========================================="
echo "All mixed graph training completed!"
echo "Date: $(date)"
echo "=========================================="
