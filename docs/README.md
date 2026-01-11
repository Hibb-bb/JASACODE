# Mixed Graph Structure Training System

A complete system for training transformers on **multiple Bayesian Network structures** and evaluating their in-context learning abilities across different graph topologies.

## 🎯 What This Does

Trains a single transformer model on 3 different graph structures simultaneously:
- **Tree** (binary tree topology)
- **Chain** (linear dependency chain)
- **General** (complex DAG)

Then evaluates whether the model can **learn the structure and CPT parameters from context** for each graph type.

## 🚀 Quick Start

### Option 1: Run Everything (One Command)
```bash
./run_mixed_pipeline.sh 42 100 20000 50000
#                       │   │    │      └─ training steps
#                       │   │    └──────── graphs per structure
#                       │   └───────────── context length
#                       └───────────────── random seed
```

### Option 2: Step by Step

**1. Train the model:**
```bash
python train_mixed.py \
  --seed 42 \
  --context-len 100 \
  --train-size 20000 \
  --train-step 50000 \
  --batch-size 64 \
  --output-dir runs/mixed
```

**2. Evaluate the model:**
```bash
python eval_mixed.py \
  --checkpoint runs/mixed/.../best.ckpt \
  --output-dir eval_results \
  --test-size 1000 \
  --context-lens 1 2 5 10 20 50 100 200 500
```

**3. Analyze results:**
```bash
python analyze_mixed_eval.py --eval-dir eval_results
```

### Option 3: SLURM Cluster
```bash
# Train
sbatch train_mixed_job.sh

# Evaluate (after training completes)
sbatch eval_mixed_job.sh <checkpoint_path> <output_dir>
```

## 📊 What You Get

### Training Output
- Model checkpoints in `runs/mixed/`
- Training logs (loss, accuracy)
- TensorBoard logs

### Evaluation Output
- `eval_tv_tree.csv` - Tree structure results
- `eval_tv_chain.csv` - Chain structure results  
- `eval_tv_general.csv` - General structure results

### Analysis Output
- `tv_vs_context_all_structures.png` - Performance comparison
- `tv_per_target.png` - Per-node analysis
- `tv_heatmap.png` - Context length heatmap
- `summary_statistics.txt` - Detailed statistics

## 📖 Documentation

- **[ARCHITECTURE.md](ARCHITECTURE.md)** - System design and diagrams
- **[USAGE_GUIDE.md](USAGE_GUIDE.md)** - Detailed usage instructions
- **[DEVELOPMENT.md](DEVELOPMENT.md)** - Development notes and checklist

## 🔑 Key Features

### 1. Balanced Multi-Structure Training
Each batch contains samples from all 3 structures:
```
Batch (12 graphs) = 4 Tree + 4 Chain + 4 General
```

### 2. Per-Structure Evaluation
Critical insight: Use **one checkpoint**, test on **each structure separately**:
```
Trained Model → Test on Tree    → eval_tv_tree.csv
             → Test on Chain   → eval_tv_chain.csv
             → Test on General → eval_tv_general.csv
```

### 3. In-Context Learning Assessment
- Tests if model learns structure from context
- Measures Total Variation (TV): |predicted_prob - true_prob|
- Good model: TV decreases as context increases

## 🏗️ Project Structure

```
JASACODE/
├── data/
│   ├── mixed_graphs.py          # 3 graph structures (5 nodes each)
│   └── mixed_dataset.py         # Balanced dataset
├── train_mixed.py               # Training script
├── eval_mixed.py                # Evaluation script
├── analyze_mixed_eval.py        # Analysis & plotting
├── test_mixed_structures.py     # Unit tests
├── run_mixed_pipeline.sh        # Complete workflow
├── train_mixed_job.sh           # SLURM training job
└── eval_mixed_job.sh            # SLURM evaluation job
```

## ✅ Verification

Run unit tests to verify installation:
```bash
python test_mixed_structures.py
```

Expected output: `All tests passed! ✓`

## 📈 Expected Results

**Good Model (learning all structures):**
- TV decreases with context for all structures
- Final TV < 0.05 at high context (500)
- Similar convergence rates across structures

**Example:**
```
Structure    | Context=1 | Context=100 | Context=500
-------------|-----------|-------------|-------------
Tree         | 0.30      | 0.08        | 0.02
Chain        | 0.30      | 0.10        | 0.04
General      | 0.30      | 0.15        | 0.08
```

## 🎓 Understanding the System

### Graph Structures (5 nodes: A, B, C, D, E)

**Tree:**
```
    A (root)
   / \
  B   C
 / \
D   E
```

**Chain:**
```
A → B → C → D → E
```

**General:**
```
A ──┐
↓   │
B   │
↓   ↓
C → E
↓
D
```

### Training Data Format

Each sample: `(L, N+1)` where L=context_len+1, N=5 nodes
- **Rows 0 to L-2**: Context examples (future nodes masked)
- **Row L-1**: Test example (target + future masked)
- **Last column**: Target index (which node to predict)

### Evaluation Metric

**Total Variation (TV):** `|p_predicted - p_true|`
- Measures distance between predicted and true probability
- Range: [0, 1]
- Lower is better (0 = perfect)

## 🤝 Contributing

For development notes, see [DEVELOPMENT.md](DEVELOPMENT.md).

## 📝 Citation

If you use this code, please cite:
```bibtex
@misc{mixed_graph_icl,
  title={In-Context Learning for Mixed Graph Structures},
  author={Your Name},
  year={2026}
}
```

## 📧 Contact

For questions or issues, please open a GitHub issue.
