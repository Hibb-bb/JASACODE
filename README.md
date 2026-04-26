# Transformers Simulate MLE for Sequence Generation in Bayesian Networks

This repository contains code to reproduce the experiments in "Transformers Simulate MLE for Sequence Generation in Bayesian Networks". The results in the paper for mix-structure training (Random DAG) experiments can be reproduced via the following instructions.
For single-structure training, prediction-task training, and Sachs protein signaling, please refer to the `main` branch.

## Environment Setup

```bash
uv venv .venv
source .venv/bin/activate
uv sync
```

## Experiments

### Random DAG Training (Mixed-structure)

Both experiments train one model per seed on
Erdős–Rényi DAGs with edge probability `p ∈ [0.7, 1.0]`, context length 500,
50k steps. After each array completes, submit the matching plotting job.

### 7 nodes (3 seeds)

```bash
sh train_random_dag_7node_dense_job.sh    # seeds {1111, 2222, 3333}
sh plot_random_dag_7node_dense_job.sh     
```

Outputs: `runs/best/random_dag_7nodes_p0.7to1.0_ctx500/`

### 10 nodes (5 seeds)

```bash
sh train_random_dag_10node_dense_job.sh         # {1111, 2222, 3333, 4444, 5555}
sh plot_random_dag_10node_dense_5seeds_job.sh   
```

Outputs: `runs/best/random_dag_10nodes_p0.7to1.0_ctx500/`

## Outputs

Each plotting job produces, per run directory:

- `final_eval_random_dags.png` — held-out random DAGs (Transformer / Bayesian / Naive / averaged across nodes), mean ± std over seeds.
- `final_eval_fixed_{tree,chain,general}.png` — generalization to fixed structures.
- `training_loss_agg.png` — aggregated training loss across seeds.
