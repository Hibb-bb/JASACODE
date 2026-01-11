## Setup

```bash
uv venv .venv
source .venv/bin/activate
uv sync
```

## Usage

**Local:**
```bash

# python train.py --warmup-steps 2000 --min-lr 1e-6 --init-lr 3e-4

python train.py \
  --batch-size 64 \
  --context-len 50 \
  --graph tree \
  --train-step 50000 \
  --init-lr 1e-4 \
  --train-size 20000 \
  --test-size 5000 \
  --output-dir runs/ \
  --warmup-steps 2000 \
  --min-lr 1e-6 \
  --init-lr 3e-4 \
  --seed 42 
```

The graph argument can be in ['tree', 'general', 'chain'] 

If ```num_example=100```:

  - train size refers to the number of graph parameters we sample from (how many tables)

  - test size refers to the number of observations we evaluate on a single graph (we will have 100 * test size of graph observations for each node like MNIST)

The real training dataset size depends only on <batch-size> x <train-step>

**SLURM:**
```bash
sbatch train_job.sh
```
