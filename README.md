# Transformers Simulate MLE for Sequence Generation in Bayesian Networks

This repository contains code to reproduce the experiments in "Transformers Simulate MLE for Sequence Generation in Bayesian Networks". The results in the paper for single-structure training, prediction-task training, and Sachs protein signaling experiments can be reproduced via the following instructions.
For mixed-structure training (Random DAG), please refer to the `mixed-graph-structure` branch.

## Environment Setup

```bash
uv init
uv venv .venv
source .venv/bin/activate
uv sync
```

## Experiments

### Single Structure Training


- Single structure script <br>
```bash
sh single-structure.sh
```

- Prediction task script <br>
```bash
sh pred.sh
```

- Sachs task script <br>
```bash
sh run_sachs.sh
```

results will be stored in the `runs/` folder for each random seed, `quick_plot.py` will aggregate experiments across seeds (see below).
For a single run, 10k steps training takes 5 minutes to finish. The overall process takes roughly 10 minutes including evaluation.

All simulations are done on a single NVIDIA-A100 GPU.

- Training script with dynamic sequence length

```bash
  python train.py \
    --batch-size 16 \
    --min-context-len 50 \
    --max-context-len 500 \
    --graph chain \
    --train-step 10000 \
    --init-lr 1e-4 \
    --train-size 20000 \
    --test-size 5000 \
    --output-dir runs/ \
    --warmup-steps 2000 \
    --min-lr 1e-6 \
    --init-lr 3e-4 \
    --seed 1111
```

- Training script with fixed sequence length

```bash
  python3 train.py \
    --batch-size 64 \
    --context-len 500 \
    --graph chain \
    --train-step 1000 \
    --init-lr 1e-4 \
    --train-size 20000 \
    --test-size 5000 \
    --output-dir runs/ \
    --warmup-steps 2000 \
    --min-lr 1e-6 \
    --init-lr 3e-4 \
    --seed 1234
```

___

### Custom graph structure

To train transformers on custom binary network structure, see `data/graphs.py`

An example of creating sprinkler network

```python
def get_sprinkler(seed=2000):

    rng = np.random.default_rng(seed)
    bn = BinaryBayesNet()
    for n in ["Cloudy", "Sprinkler", "Rain", "Wet Grass"]:
        bn.add_node(n)

    bn.add_edge("Cloudy", "Sprinkler")
    bn.add_edge("Cloudy", "Rain")


    bn.add_edge("Sprinkler", "Wet Grass")
    bn.add_edge("Rain", "Wet Grass")

    bn.set_parents("Cloudy", [])
    bn.set_parents("Sprinkler", ["Cloudy"])
    bn.set_parents("Rain", ["Cloudy"])
    bn.set_parents("Wet Grass", ["Sprinkler", "Rain"])

    # Random CPTs
    bn.set_cpt("Cloudy", random_binary_cpt(0, rng))
    bn.set_cpt("Sprinkler", random_binary_cpt(1, rng))
    bn.set_cpt("Rain", random_binary_cpt(1, rng))
    bn.set_cpt("Wet Grass", random_binary_cpt(2, rng))

    return bn
```

Next, modify `train.py` so it imports `get_sprinkler` from `data/graphs.py`.

## Outputs

| File | Use |
|------|-----|
| `quick_plot.py` | Single structure Mean ± std over five seeds from eval CSVs → figure under `imgs/`. |
| `quick_plot_sachs.py` | Same style for real-Sachs eval outputs (e.g. under `runs/sachs_real_eval/…`). |
| `plot_loss.py` | Loss and TV from Lightning `metrics.csv`. |
| `plot_loss_runs.sh` | Calls `plot_loss.py` for tree / chain / general paths aligned with `pred.sh` defaults. |

___

## Rebuttal experiments (theory architecture + prediction objective)

Two additions address the second-round review:

1. **Theorem 4.1 architecture.** `--arch theory` in `train.py` / `train_pred.py` selects
   `TheoryTransformerBinaryHead` (`model/models.py`): one-hot input, width `(2N+1)d` by default,
   two layers of single-head softmax attention (token never attends to itself) + square ReLU
   feed-forward, residual connections, no LayerNorm / dropout / positional embedding, linear readout.
   `--n-head`, `--n-embd`, `--num-layers` also apply to the GPT-2 block (`--arch gpt2`, default).
2. **Prediction objective.** `train_pred.py` trains with cross-entropy on the realized target value.
   `--cpt-prior uniform` (default) uses the same U(0.1,0.9) CPT prior as the main experiments;
   `--seeds` and `--no-plot` allow one seed per SLURM task.

Note: the GPT-2 "non-causal" patch now passes an explicit 4D all-zeros mask; with
`transformers>=4.4x` flipping `attn.is_causal` alone leaves the model causal.

```bash
python slurm/make_configs.py                                   # writes slurm/configs/*.txt
sbatch --array=1-15 slurm/array.sbatch slurm/configs/A2.txt     # one (graph, seed) per task
python plot_rebuttal.py                                        # -> imgs/rebuttal/, summary.csv
```

| config | script | model | loss |
|---|---|---|---|
| ref | train.py | gpt2 L4 H8 D256 | TV |
| A1 | train.py | gpt2 L2 H1 D256 | TV |
| A2 (+w64, w256) | train.py | theory L2 H1 | TV |
| B | train_pred.py | gpt2 L4 H8 D256 | cross-entropy |
| C1 | train_pred.py | gpt2 L2 H1 D256 | cross-entropy |
| C2 (+w256) | train_pred.py | theory L2 H1 | cross-entropy |

Outputs land in `runs_rebuttal/{tv,bce}/<graph>/seed_<s>/<ctx>/<model_tag>/...`.
