source .venv/bin/activate

NUM_NODES=10
EDGE_PROB_MIN=0.7
EDGE_PROB_MAX=1.0
OUTPUT_DIR=runs/best/random_dag_${NUM_NODES}nodes_p${EDGE_PROB_MIN}to${EDGE_PROB_MAX}_ctx500
RUN_SUFFIX="${NUM_NODES}nodes_p=${EDGE_PROB_MIN}to${EDGE_PROB_MAX}_ctx500"

echo "=========================================="
echo "Replotting with 5 seeds"
echo "Date: $(date)"
echo "=========================================="

echo "Plotting random DAG eval..."
python plot_eval_random_dag.py \
  --output-dir $OUTPUT_DIR \
  --run-prefix rdag \
  --run-suffix "$RUN_SUFFIX" \
  --eval-type random_dags \
  --num-nodes $NUM_NODES \
  --seeds 1111 2222 3333 4444 5555

for structure in tree chain general; do
  echo "Plotting fixed structure: $structure ..."
  python plot_eval_random_dag.py \
    --output-dir $OUTPUT_DIR \
    --run-prefix rdag \
    --run-suffix "$RUN_SUFFIX" \
    --eval-type fixed \
    --structure $structure \
    --num-nodes $NUM_NODES \
    --seeds 1111 2222 3333 4444 5555
done

echo "Plotting aggregated training loss..."
python plot_training_loss_agg.py \
  --output-dir $OUTPUT_DIR \
  --title "Random DAG — ${NUM_NODES} Nodes Dense — Training Loss"

echo ""
echo "=========================================="
echo "All plots done!  $(date)"
echo "=========================================="
