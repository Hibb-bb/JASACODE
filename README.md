# JASA — Random DAG In-Context Learning

Train a non-causal GPT-2 on random DAG Bayesian networks and evaluate against
naive / Bayesian baselines. This repo currently keeps only the 7-node and
10-node dense random-DAG experiments.

## Setup

```bash
uv venv .venv
source .venv/bin/activate
uv sync
```

## Experiments

Both experiments train one model per seed (parallel SLURM job array) on
Erdős–Rényi DAGs with edge probability `p ∈ [0.7, 1.0]`, context length 500,
50k steps. After each array completes, submit the matching plotting job.

### 7 nodes (3 seeds)

```bash
sbatch train_random_dag_7node_dense_job.sh    # array 0–2, seeds {1111, 2222, 3333}
sbatch plot_random_dag_7node_dense_job.sh     # after the array finishes
```

Outputs: `runs/best/random_dag_7nodes_p0.7to1.0_ctx500/`

### 10 nodes (5 seeds)

```bash
sbatch train_random_dag_10node_dense_job.sh         # array 0–4, seeds {1111…5555}
sbatch plot_random_dag_10node_dense_5seeds_job.sh   # after the array finishes
```

Outputs: `runs/best/random_dag_10nodes_p0.7to1.0_ctx500/`

## Outputs

Each plotting job produces, per run directory:

- `final_eval_random_dags.png` — held-out random DAGs (Transformer / Bayesian / Naive / averaged across nodes), mean ± std over seeds.
- `final_eval_fixed_{tree,chain,general}.png` — generalization to fixed structures.
- `training_loss_agg.png` — aggregated training loss across seeds.
