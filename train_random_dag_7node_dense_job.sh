#!/bin/bash
#SBATCH --account=p32626
#SBATCH --job-name=rdag_7n_dense
#SBATCH --array=0-2
#SBATCH --nodes=1
#SBATCH --output=/projects/p32626/JASACODE/slurm_log/slurm_rdag7dense_%A_%a.out
#SBATCH --error=/projects/p32626/JASACODE/slurm_log/slurm_rdag7dense_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=b0976960890@gmail.com

cd /projects/p32626/JASACODE
source .venv/bin/activate

SEEDS=(1111 2222 3333)
seed=${SEEDS[$SLURM_ARRAY_TASK_ID]}

NUM_NODES=7
EDGE_PROB_MIN=0.7
EDGE_PROB_MAX=1.0
OUTPUT_DIR=runs/best/random_dag_${NUM_NODES}nodes_p${EDGE_PROB_MIN}to${EDGE_PROB_MAX}_ctx500

echo "=========================================="
echo "Random DAG Training — 7 Nodes, Dense (p 0.7–1.0)"
echo "Array task $SLURM_ARRAY_TASK_ID / seed=$seed"
echo "=========================================="
echo "Job ID: $SLURM_ARRAY_JOB_ID  Task: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_NODELIST"
echo "Date: $(date)"
echo "=========================================="

CUDA_LAUNCH_BLOCKING=1 python train_random_dag.py \
  --batch-size 16 \
  --num-nodes $NUM_NODES \
  --edge-prob-min $EDGE_PROB_MIN \
  --edge-prob-max $EDGE_PROB_MAX \
  --context-len 500 \
  --train-step 50000 \
  --init-lr 3e-4 \
  --warmup-steps 2000 \
  --min-lr 1e-6 \
  --test-size 5000 \
  --num-eval-dags 20 \
  --output-dir $OUTPUT_DIR \
  --seed $seed

echo "Completed training + eval: seed=$seed"

echo "Generating training loss plot for seed=$seed"
python plot_training_loss.py \
  --metrics_csv ${OUTPUT_DIR}/rdag_seed${seed}_${NUM_NODES}nodes_p=${EDGE_PROB_MIN}to${EDGE_PROB_MAX}_ctx500/logs/version_0/metrics.csv \
  -o ${OUTPUT_DIR}/rdag_seed${seed}_${NUM_NODES}nodes_p=${EDGE_PROB_MIN}to${EDGE_PROB_MAX}_ctx500/training_loss_plot.png

echo "Done: seed=$seed  $(date)"
