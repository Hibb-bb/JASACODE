#!/bin/bash
#SBATCH --account=p32626
#SBATCH --job-name=rdag_5n
#SBATCH --nodes=1
#SBATCH --output=/projects/p32626/JASACODE/slurm_log/slurm_rdag_%j.out
#SBATCH --error=/projects/p32626/JASACODE/slurm_log/slurm_rdag_%j.err
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

echo "=========================================="
echo "Random DAG Training"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Date: $(date)"
echo "=========================================="

NUM_NODES=5
EDGE_PROB_MIN=0.1
EDGE_PROB_MAX=0.8
OUTPUT_DIR=runs/best/random_dag_${NUM_NODES}nodes_p${EDGE_PROB_MIN}to${EDGE_PROB_MAX}_ctx500

for seed in 1111 2222 3333 4444 5555; do

  echo ""
  echo "----------------------------------------"
  echo "Training random DAG: seed=$seed"
  echo "----------------------------------------"

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

  echo "Completed training: seed=$seed"

  # Generate training loss plot
  echo "Generating training loss plot for seed=$seed"
  python plot_training_loss.py \
    --metrics_csv ${OUTPUT_DIR}/rdag_seed${seed}_${NUM_NODES}nodes_p=${EDGE_PROB_MIN}to${EDGE_PROB_MAX}_ctx500/logs/version_0/metrics.csv \
    -o ${OUTPUT_DIR}/rdag_seed${seed}_${NUM_NODES}nodes_p=${EDGE_PROB_MIN}to${EDGE_PROB_MAX}_ctx500/training_loss_plot.png

  echo "Completed all tasks: seed=$seed"

done

echo ""
echo "=========================================="
echo "All seeds done — generating aggregate plots (mean +/- std across seeds)"
echo "=========================================="

RUN_SUFFIX="${NUM_NODES}nodes_p=${EDGE_PROB_MIN}to${EDGE_PROB_MAX}_ctx500"

# Plot random DAG eval results
echo "Plotting random DAG eval..."
python plot_eval_random_dag.py \
  --output-dir $OUTPUT_DIR \
  --run-prefix rdag \
  --run-suffix "$RUN_SUFFIX" \
  --eval-type random_dags \
  --num-nodes $NUM_NODES

# Plot fixed structure generalization (tree, chain, general)
for structure in tree chain general; do
  echo "Plotting fixed structure: $structure ..."
  python plot_eval_random_dag.py \
    --output-dir $OUTPUT_DIR \
    --run-prefix rdag \
    --run-suffix "$RUN_SUFFIX" \
    --eval-type fixed \
    --structure $structure \
    --num-nodes $NUM_NODES
done

echo "Plotting aggregated training loss..."
python plot_training_loss_agg.py \
  --output-dir $OUTPUT_DIR \
  --title "Random DAG — ${NUM_NODES} Nodes — Training Loss"

echo ""
echo "=========================================="
echo "All random DAG training completed!"
echo "Date: $(date)"
echo "=========================================="
