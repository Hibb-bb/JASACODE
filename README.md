## Setup

```bash
uv venv .venv
source .venv/bin/activate
uv sync
```

## Usage

**Local:**
```bash
python train.py \
  --batch-size 64 \
  --context-len 100 \
  --graph tree \
  --train-step 50000 \
  --init-lr 1e-4 \
  --train-size 10000 \
  --test-size 1000 \
  --output-dir runs/ \
  --seed 42
```

The graph argument can be in ['tree', 'general', 'chain'] 

train size refers to the number of graph parameters we sample from

test size refers to the number of observations we evaluate on a single graph

The real training dataset size depends only on <batch-size> x <train-step>

**SLURM:**
```bash
sbatch train_job.sh
```
