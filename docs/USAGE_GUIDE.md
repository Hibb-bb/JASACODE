# Usage Guide

Complete guide to using the Mixed Graph Structure Training System.

## Table of Contents

1. [Installation](#installation)
2. [Training](#training)
3. [Evaluation](#evaluation)
4. [Analysis](#analysis)
5. [Job Scripts (SLURM)](#job-scripts-slurm)
6. [Parameter Tuning](#parameter-tuning)
7. [Interpreting Results](#interpreting-results)
8. [Troubleshooting](#troubleshooting)

## Installation

### 1. Clone Repository
```bash
cd /path/to/your/workspace
git clone <repository-url>
cd JASACODE
```

### 2. Setup Environment
```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -r requirements.txt
```

### 3. Verify Installation
```bash
python test_mixed_structures.py
```

Expected output: `All tests passed! ✓`

## Training

### Basic Training

**Local (CPU/GPU):**
```bash
python train_mixed.py \
  --seed 42 \
  --context-len 100 \
  --train-size 20000 \
  --train-step 50000 \
  --batch-size 64 \
  --output-dir runs/mixed
```

**SLURM Cluster:**
```bash
sbatch train_mixed_job.sh
```

### Training Parameters

#### Required Parameters
None - all have defaults

#### Important Parameters

**`--seed`** (default: 1111)
- Random seed for reproducibility
- Different seeds → different CPT samples

**`--context-len`** (default: 100)
- Number of context examples per sample
- Higher → more in-context learning signal
- Typical values: 10, 20, 50, 100, 200

**`--train-size`** (default: 20000)
- Number of graph instances per structure
- Total graphs = train_size × 3
- Typical values: 10000, 20000, 50000

**`--train-step`** (default: 50000)
- Optimization steps
- More steps → better convergence
- Typical values: 20000, 50000, 100000

**`--batch-size`** (default: 64)
- Total graphs per batch
- Should be divisible by 3 for even split
- Typical values: 12, 24, 48, 64, 96

**`--output-dir`** (default: outputs_mixed)
- Directory for checkpoints and logs
- Creates subdirectory with run details

#### Model Architecture Parameters

**`--num-layer`** (default: 6)
- Number of transformer layers
- More layers → more capacity

**`--num-head`** (default: 8)
- Number of attention heads
- Typical values: 4, 8, 16

**`--embedding-dim`** (default: 256)
- Embedding dimension
- Larger → more capacity, slower

**`--dropout`** (default: 0.1)
- Dropout rate for regularization

**`--lr`** (default: 1e-4)
- Learning rate
- Adjust if training unstable

### Training Output

**Directory structure:**
```
runs/mixed/mixed_seed42_ctx100_train20000/
├── logs/
│   └── lightning_logs/
│       └── version_0/
│           ├── checkpoints/
│           │   ├── best.ckpt        ← Use this for evaluation
│           │   └── last.ckpt
│           ├── events.out.tfevents  ← TensorBoard logs
│           └── hparams.yaml
└── train_config.txt
```

**Monitor training:**
```bash
# TensorBoard
tensorboard --logdir runs/mixed

# Or check logs
cat runs/mixed/.../train_config.txt
```

### Example Training Commands

**Quick test (fast, small):**
```bash
python train_mixed.py \
  --seed 1 \
  --context-len 10 \
  --train-size 1000 \
  --train-step 5000 \
  --batch-size 12 \
  --output-dir test_run
```

**Standard experiment:**
```bash
python train_mixed.py \
  --seed 42 \
  --context-len 100 \
  --train-size 20000 \
  --train-step 50000 \
  --batch-size 64 \
  --output-dir runs/mixed
```

**Large-scale experiment:**
```bash
python train_mixed.py \
  --seed 1111 \
  --context-len 200 \
  --train-size 50000 \
  --train-step 100000 \
  --batch-size 96 \
  --num-layer 12 \
  --embedding-dim 512 \
  --output-dir runs/mixed_large
```

## Evaluation

### Basic Evaluation

**Find your checkpoint:**
```bash
# Example path
CKPT="runs/mixed/mixed_seed42_ctx100_train20000/logs/lightning_logs/version_0/checkpoints/best.ckpt"
```

**Run evaluation:**
```bash
python eval_mixed.py \
  --checkpoint $CKPT \
  --output-dir eval_results \
  --test-size 1000 \
  --context-lens 1 2 5 10 20 50 100 200 300 400 500
```

**SLURM:**
```bash
sbatch eval_mixed_job.sh $CKPT eval_results
```

### Evaluation Parameters

**`--checkpoint`** (required)
- Path to trained model checkpoint (.ckpt file)
- Use `best.ckpt` for best validation performance

**`--output-dir`** (default: eval_results)
- Where to save CSV files
- Creates subdirectory with run details

**`--test-size`** (default: 1000)
- Number of episodes per context length
- More episodes → smoother curves
- Typical values: 500, 1000, 2000

**`--context-lens`** (default: [1,2,5,10,20,50,100,200,300,400,500])
- List of context lengths to test
- Space-separated integers
- Example: `--context-lens 1 10 100 500`

**`--batch-size`** (default: 512)
- Batch size for inference
- Larger → faster (if GPU memory allows)

**`--eval-seed`** (default: 9999)
- Random seed for evaluation
- Different from training seed
- Ensures evaluation on unseen graphs

### Evaluation Output

**Directory structure:**
```
eval_results/
├── eval_tv_tree.csv       ← Tree structure results
├── eval_tv_chain.csv      ← Chain structure results
└── eval_tv_general.csv    ← General structure results
```

**CSV columns:**
- `context_len`: Number of context examples
- `target_index`: Which node (0=A, 1=B, 2=C, 3=D, 4=E)
- `episode`: Episode number (0 to test_size-1)
- `p_hat`: Model's predicted probability
- `p_true`: True probability from CPT
- `tv`: Total variation = |p_hat - p_true|
- `y_test`: Actual observed value (0 or 1)
- `parents_cfg`: Parent configuration encoding

### Example Evaluation Commands

**Quick test:**
```bash
python eval_mixed.py \
  --checkpoint $CKPT \
  --output-dir eval_quick \
  --test-size 100 \
  --context-lens 1 10 100
```

**Standard:**
```bash
python eval_mixed.py \
  --checkpoint $CKPT \
  --output-dir eval_results \
  --test-size 1000 \
  --context-lens 1 2 5 10 20 50 100 200 500
```

**Comprehensive:**
```bash
python eval_mixed.py \
  --checkpoint $CKPT \
  --output-dir eval_comprehensive \
  --test-size 5000 \
  --context-lens 1 2 3 4 5 10 15 20 30 50 75 100 150 200 300 500
```

## Analysis

### Basic Analysis

```bash
python analyze_mixed_eval.py --eval-dir eval_results
```

This generates:
- `tv_vs_context_all_structures.png`
- `tv_per_target.png`
- `tv_heatmap.png`
- `summary_statistics.txt`

### Analysis Parameters

**`--eval-dir`** (required)
- Directory containing eval_tv_*.csv files
- Same as evaluation --output-dir

**`--output-dir`** (optional)
- Where to save plots (default: same as eval-dir)

### Understanding the Outputs

#### 1. TV vs Context (all structures)

**What it shows:**
- X-axis: Context length (log scale)
- Y-axis: Mean Total Variation
- One line per structure
- Shaded area: ±1 standard error

**Good result:**
- All lines decrease
- Lines converge to low TV (< 0.05)
- Similar slopes for all structures

**Bad result:**
- Flat lines (no learning)
- High final TV (> 0.15)
- Large differences between structures

#### 2. Per-Target Analysis

**What it shows:**
- One subplot per structure
- Multiple lines (one per node A-E)
- Shows which nodes are easier/harder

**Interpretation:**
- Root nodes (no parents) should be easy
- Leaf nodes (many parents) may be harder
- Inconsistent curves → model confusion

#### 3. Heatmap

**What it shows:**
- Rows: Structures (tree, chain, general)
- Columns: Context lengths
- Color: TV value (green=good, red=bad)

**Interpretation:**
- Green column → good at that context
- Red row → structure is difficult
- Smooth gradient → consistent learning

#### 4. Summary Statistics

**Contents:**
- Overall TV statistics per structure
- Per context length breakdown
- Per target node breakdown
- Cross-structure comparison

**Key metrics:**
- Mean TV: Overall performance
- Final TV (ctx=500): Convergence quality
- Min/Max TV: Variability

## Job Scripts (SLURM)

### Training Job

**File:** `train_mixed_job.sh`

**Usage:**
```bash
sbatch train_mixed_job.sh
```

**Customize:**
Edit the script to change:
- `#SBATCH` directives (time, memory, GPUs)
- Training parameters (seed, context-len, etc.)

**Example customization:**
```bash
#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1

python train_mixed.py \
  --seed 42 \
  --context-len 200 \
  --train-size 50000 \
  --train-step 100000 \
  --batch-size 96
```

### Evaluation Job

**File:** `eval_mixed_job.sh`

**Usage:**
```bash
sbatch eval_mixed_job.sh <checkpoint> <output_dir>
```

**Example:**
```bash
sbatch eval_mixed_job.sh \
  runs/mixed/.../best.ckpt \
  eval_results_seed42
```

**Parameters:**
- `$1`: Path to checkpoint
- `$2`: Output directory name

### Complete Pipeline

**File:** `run_mixed_pipeline.sh`

**Usage:**
```bash
./run_mixed_pipeline.sh <seed> <context_len> <train_size> <train_step>
```

**Example:**
```bash
./run_mixed_pipeline.sh 42 100 20000 50000
```

**What it does:**
1. Trains model
2. Waits for completion
3. Runs evaluation
4. Generates analysis plots

## Parameter Tuning

### For Better Performance

**If model not learning (TV not decreasing):**
1. Increase `--train-step` (more optimization)
2. Increase `--train-size` (more data diversity)
3. Increase `--num-layer` or `--embedding-dim` (more capacity)
4. Decrease `--lr` if training unstable
5. Increase `--context-len` (more signal per sample)

**If training too slow:**
1. Decrease `--batch-size` (if memory limited)
2. Decrease `--train-size` (fewer unique graphs)
3. Decrease `--num-layer` (simpler model)
4. Use fewer `--context-lens` in evaluation

**If one structure performs poorly:**
1. Check graph definition in `data/mixed_graphs.py`
2. Increase `--train-step` (needs more time)
3. Adjust batch distribution in dataset

### Recommended Configurations

**Fast prototyping:**
```bash
--context-len 10
--train-size 1000
--train-step 5000
--batch-size 12
```

**Standard experiment:**
```bash
--context-len 100
--train-size 20000
--train-step 50000
--batch-size 64
```

**Publication quality:**
```bash
--context-len 200
--train-size 50000
--train-step 100000
--batch-size 96
--num-layer 12
--embedding-dim 512
```

## Interpreting Results

### Expected Patterns

**Good Model:**
```
Structure  | Ctx=1  | Ctx=10 | Ctx=100 | Ctx=500
-----------|--------|--------|---------|--------
Tree       | 0.30   | 0.15   | 0.05    | 0.02
Chain      | 0.30   | 0.18   | 0.08    | 0.04
General    | 0.30   | 0.22   | 0.12    | 0.08
```
✓ TV decreases consistently
✓ Final TV < 0.10
✓ General hardest (most complex)

**Moderate Model:**
```
Structure  | Ctx=1  | Ctx=10 | Ctx=100 | Ctx=500
-----------|--------|--------|---------|--------
Tree       | 0.30   | 0.20   | 0.12    | 0.10
Chain      | 0.30   | 0.22   | 0.15    | 0.12
General    | 0.30   | 0.25   | 0.20    | 0.18
```
~ TV decreases but not enough
~ Final TV: 0.10-0.20
~ May need more training

**Poor Model:**
```
Structure  | Ctx=1  | Ctx=10 | Ctx=100 | Ctx=500
-----------|--------|--------|---------|--------
Tree       | 0.30   | 0.28   | 0.26    | 0.25
Chain      | 0.30   | 0.29   | 0.28    | 0.27
General    | 0.30   | 0.30   | 0.30    | 0.30
```
✗ Minimal improvement
✗ High final TV (> 0.20)
✗ Not learning from context

### Structure-Specific Insights

**Tree performing best:**
- Hierarchical structure is easier
- Clear parent-child relationships
- Lower depth → faster learning

**Chain in middle:**
- Sequential dependencies
- Moderate complexity
- May need more context for tail nodes

**General performing worst:**
- Multiple parents per node
- More complex interactions
- Needs more capacity/data

## Troubleshooting

### Training Issues

**Error: "CUDA out of memory"**
- Solution: Reduce `--batch-size`
- Or: Reduce `--context-len`
- Or: Use gradient accumulation

**Error: "No module named 'data.mixed_graphs'"**
- Solution: Ensure in JASACODE directory
- Check: `data/__init__.py` has proper exports

**Training loss not decreasing:**
- Check: Learning rate (try 1e-3 or 5e-5)
- Check: Data loading (print batch shapes)
- Check: Model capacity (increase layers/dim)

**Training very slow:**
- Check: GPU utilization (`nvidia-smi`)
- Reduce: batch_size or context_len
- Check: Data loading bottleneck

### Evaluation Issues

**Error: "Checkpoint not found"**
- Check: Path is correct
- Use: Full absolute path
- Check: best.ckpt exists

**TV values all ~0.30 (random guessing):**
- Model didn't train properly
- Check training loss/accuracy
- May need more train_step

**One structure has NaN:**
- Check: Graph definition valid DAG
- Check: CPT probabilities valid [0,1]
- Check: No numerical instability

### Analysis Issues

**Error: "No CSV files found"**
- Check: eval-dir path correct
- Ensure: eval_tv_*.csv files exist
- Check: Evaluation completed successfully

**Plots look wrong:**
- Check: CSV data looks reasonable
- Try: Different context_lens
- Check: test_size sufficient (>100)

### Common Pitfalls

**Mistake:** Using same seed for train and eval
- Fix: Use different `--eval-seed` in evaluation

**Mistake:** Batch size not divisible by 3
- Fix: Use batch_size = 12, 24, 48, 64, 96, etc.

**Mistake:** Comparing checkpoints with different train_size
- Fix: Keep train_size consistent for fair comparison

**Mistake:** Not enough test episodes
- Fix: Use test_size >= 500 for smooth curves

## Advanced Usage

### Custom Graph Structures

1. Add function to `data/mixed_graphs.py`:
```python
def get_custom_5node(seed=2000):
    bn = BinaryBayesNet()
    # Define your structure
    ...
    return bn
```

2. Update `get_mixed_graph_structures()`:
```python
def get_mixed_graph_structures(seed=2000):
    return [
        get_tree_5node(seed),
        get_chain_5node(seed),
        get_general_5node(seed),
        get_custom_5node(seed + 3),
    ]
```

3. Train and evaluate as usual

### Multiple Random Seeds

```bash
for seed in 1111 2222 3333 4444 5555; do
    python train_mixed.py --seed $seed --output-dir runs/seed_$seed
    CKPT="runs/seed_$seed/.../best.ckpt"
    python eval_mixed.py --checkpoint $CKPT --output-dir eval_seed_$seed
done
```

### Comparing Models

```bash
# Train two models
python train_mixed.py --context-len 50 --output-dir runs/ctx50
python train_mixed.py --context-len 200 --output-dir runs/ctx200

# Evaluate both
python eval_mixed.py --checkpoint runs/ctx50/.../best.ckpt --output-dir eval_ctx50
python eval_mixed.py --checkpoint runs/ctx200/.../best.ckpt --output-dir eval_ctx200

# Compare results
python analyze_mixed_eval.py --eval-dir eval_ctx50
python analyze_mixed_eval.py --eval-dir eval_ctx200
```

## Next Steps

- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Understand system design
- **[DEVELOPMENT.md](DEVELOPMENT.md)** - Contribute to codebase
- **[README.md](README.md)** - Project overview

## Support

For issues or questions:
1. Check this guide
2. Check code comments
3. Run unit tests: `python test_mixed_structures.py`
4. Open GitHub issue with:
   - Command you ran
   - Error message
   - Expected vs actual behavior
