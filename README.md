## Environment Setup

```bash
uv init
uv venv .venv
source .venv/bin/activate
uv sync
```

## Usage

### Standard Training (Fig. 2 to Fig. 4)


- 7 Node configuration <br>
```bash
sh 7-node.sh
```

- 5 Node configuration <br>
```bash
sh 5-node.sh
```
results will be stored in the `runs/` folder for each random seed, `quick_plot.py` will aggregate experiments across seeds.
For a single run, 10k steps training takes 5 minutes to finish. The overall process takes roughly 10 minutes including evaluation.

All simulations are done on a single NVIDIA-A100 GPU.

- Loss curve visualization example <br>
```bash
python3 plot_loss.py --metrics_csv runs/tree5/seed_1234/50to500/20000/logs/version_0/metrics.csv --output ./loss.png
```

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
  python train.py \
    --batch-size 16 \
    --context-len 100 \
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

```
# salloc -p debug_a100 -t 02:00:00 --gres=gpu:1
# srun --pty bash -l



# salloc -p debug -t 02:00:00
```


```uv run python train_sachs.py --train-step 50000 --batch-size 16 --context-len 200```