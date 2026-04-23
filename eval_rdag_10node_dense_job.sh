#!/bin/bash
#SBATCH --account=p32626
#SBATCH --job-name=eval_rdag10n
#SBATCH --nodes=1
#SBATCH --output=/projects/p32626/JASACODE/slurm_log/slurm_eval_rdag10dense_%j.out
#SBATCH --error=/projects/p32626/JASACODE/slurm_log/slurm_eval_rdag10dense_%j.err
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
echo "Evaluate 10-node dense rdag on fixed structures (tree/chain/general)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Date: $(date)"
echo "=========================================="

NUM_NODES=10
CKPT_BASE=runs/best/random_dag_${NUM_NODES}nodes_p0.7to1.0_ctx500
RUN_SUFFIX="${NUM_NODES}nodes_p=0.7to1.0_ctx500"

for seed in 1111 2222 3333; do
  CKPT=${CKPT_BASE}/rdag_seed${seed}_${RUN_SUFFIX}/logs/version_0/checkpoints/best.ckpt
  EVAL_DIR=${CKPT_BASE}/rdag_seed${seed}_${RUN_SUFFIX}_eval/fixed_structures

  echo ""
  echo "--- seed=$seed ---"
  python eval_fixed_structures.py \
    --checkpoint "$CKPT" \
    --output-dir "$EVAL_DIR" \
    --test-size 5000 \
    --seed $seed \
    --num-nodes $NUM_NODES
done

echo ""
echo "Generating aggregate plots ..."

for structure in tree chain general; do
  echo "Plotting fixed structure: $structure ..."
  python plot_eval_random_dag.py \
    --output-dir "$CKPT_BASE" \
    --run-prefix rdag \
    --run-suffix "$RUN_SUFFIX" \
    --eval-type fixed \
    --structure "$structure" \
    --num-nodes $NUM_NODES \
    --seeds 1111 2222 3333
done

echo ""
echo "=========================================="
echo "Done! $(date)"
echo "=========================================="
