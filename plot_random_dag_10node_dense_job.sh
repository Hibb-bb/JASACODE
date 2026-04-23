#!/bin/bash
#SBATCH --account=p32626
#SBATCH --job-name=plot_rdag10n
#SBATCH --nodes=1
#SBATCH --output=/projects/p32626/JASACODE/slurm_log/slurm_plot_rdag10dense_%j.out
#SBATCH --error=/projects/p32626/JASACODE/slurm_log/slurm_plot_rdag10dense_%j.err
#SBATCH --time=00:10:00
#SBATCH --partition=short
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=b0976960890@gmail.com

cd /projects/p32626/JASACODE
source .venv/bin/activate

NUM_NODES=10
EDGE_PROB_MIN=0.7
EDGE_PROB_MAX=1.0
OUTPUT_DIR=runs/best/random_dag_${NUM_NODES}nodes_p${EDGE_PROB_MIN}to${EDGE_PROB_MAX}_ctx500
RUN_SUFFIX="${NUM_NODES}nodes_p=${EDGE_PROB_MIN}to${EDGE_PROB_MAX}_ctx500"

echo "=========================================="
echo "Generating aggregate plots (mean +/- std across seeds)"
echo "Date: $(date)"
echo "=========================================="

echo "Plotting random DAG eval..."
python plot_eval_random_dag.py \
  --output-dir $OUTPUT_DIR \
  --run-prefix rdag \
  --run-suffix "$RUN_SUFFIX" \
  --eval-type random_dags \
  --num-nodes $NUM_NODES \
  --seeds 1111 2222 3333

for structure in tree chain general; do
  echo "Plotting fixed structure: $structure ..."
  python plot_eval_random_dag.py \
    --output-dir $OUTPUT_DIR \
    --run-prefix rdag \
    --run-suffix "$RUN_SUFFIX" \
    --eval-type fixed \
    --structure $structure \
    --num-nodes $NUM_NODES \
    --seeds 1111 2222 3333
done

echo "Plotting aggregated training loss..."
python plot_training_loss_agg.py \
  --output-dir $OUTPUT_DIR \
  --title "Random DAG — ${NUM_NODES} Nodes Dense — Training Loss"

echo ""
echo "=========================================="
echo "All plots done!  $(date)"
echo "=========================================="
