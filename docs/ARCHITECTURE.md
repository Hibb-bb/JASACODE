# System Architecture

## Overview

The Mixed Graph Structure Training System enables a transformer to learn multiple Bayesian Network structures through in-context learning.

## System Diagram

```
╔════════════════════════════════════════════════════════════════════════════════╗
║                    MIXED GRAPH STRUCTURE TRAINING SYSTEM                        ║
╚════════════════════════════════════════════════════════════════════════════════╝

┌──────────────────────────────────────────────────────────────────────────────┐
│                            1. GRAPH STRUCTURES (5 nodes each)                 │
└──────────────────────────────────────────────────────────────────────────────┘

   Tree Structure:            Chain Structure:          General Structure:
                                                  
        A (root)                   A                      A ──┐
       / \                         ↓                      ↓   │
      B   C (depth-1)              B                      B   │
     / \                           ↓                      ↓   ↓
    D   E (depth-2)                C                      C → E
                                   ↓                      ↓
                                   D                      D
                                   ↓
                                   E

    4 edges                     4 edges                   6 edges
    Max depth: 2                Linear chain              Complex DAG

┌──────────────────────────────────────────────────────────────────────────────┐
│                          2. TRAINING DATA GENERATION                          │
└──────────────────────────────────────────────────────────────────────────────┘

For each batch (B=12 graphs):

┌─────────────────┬─────────────────┬─────────────────┐
│  Tree (4)       │  Chain (4)      │  General (4)    │
├─────────────────┼─────────────────┼─────────────────┤
│ graph_id: 0 to  │ graph_id: 0 to  │ graph_id: 0 to  │
│  train_size-1   │  train_size-1   │  train_size-1   │
│ struct_id: 0    │ struct_id: 1    │ struct_id: 2    │
│                 │                 │                 │
│ Sample from     │ Sample from     │ Sample from     │
│ tree template   │ chain template  │ general template│
│ with CPT i      │ with CPT j      │ with CPT k      │
└─────────────────┴─────────────────┴─────────────────┘

Each sample: (L, N+1) where L = context_len + 1, N = 5
  - Row 0 to L-2: context examples (mask future nodes)
  - Row L-1: test example (mask target + future)
  - Column N: target index feature (same for all rows)

┌──────────────────────────────────────────────────────────────────────────────┐
│                               3. MODEL ARCHITECTURE                           │
└──────────────────────────────────────────────────────────────────────────────┘

Input: (B=12, L=101, D=6)
  ↓
┌─────────────────────────┐
│  Embedding Layer        │  D=6 → embedding_dim=256
└─────────────────────────┘
  ↓
┌─────────────────────────┐
│  Transformer Layers     │  6 layers, 8 heads
│  (Non-Causal)           │  Attend to all positions
└─────────────────────────┘
  ↓
┌─────────────────────────┐
│  Binary Head            │  Predict target node
└─────────────────────────┘
  ↓
Output: (B=12,) - predictions for target nodes

Loss: Binary Cross Entropy
  - Only on test row (last row)
  - Supervised by true node value

┌──────────────────────────────────────────────────────────────────────────────┐
│                            4. EVALUATION STRATEGY                             │
└──────────────────────────────────────────────────────────────────────────────┘

                        ┌─────────────────────┐
                        │  Trained Checkpoint │
                        │   (mixed_model.ckpt)│
                        └──────────┬──────────┘
                                   │
            ┌──────────────────────┼──────────────────────┐
            │                      │                      │
            ↓                      ↓                      ↓
    ┌───────────────┐      ┌───────────────┐    ┌───────────────┐
    │ Eval on TREE  │      │ Eval on CHAIN │    │ Eval on GEN   │
    │               │      │               │    │               │
    │ 1 fixed graph │      │ 1 fixed graph │    │ 1 fixed graph │
    │ Different CPT │      │ Different CPT │    │ Different CPT │
    └───────┬───────┘      └───────┬───────┘    └───────┬───────┘
            │                      │                      │
            ↓                      ↓                      ↓
    Context: 1,2,5...       Context: 1,2,5...     Context: 1,2,5...
    Episodes: 1000          Episodes: 1000        Episodes: 1000
            │                      │                      │
            ↓                      ↓                      ↓
    eval_tv_tree.csv        eval_tv_chain.csv     eval_tv_general.csv

Each CSV contains:
  - context_len: Number of context examples
  - target_index: Which node (0-4 for A-E)
  - episode: Episode number
  - p_hat: Model prediction
  - p_true: True probability (from CPT)
  - tv: Total variation |p_hat - p_true|
  - y_test: Actual observation
  - parents_cfg: Parent configuration

┌──────────────────────────────────────────────────────────────────────────────┐
│                              5. ANALYSIS PIPELINE                             │
└──────────────────────────────────────────────────────────────────────────────┘

Input: eval_tv_{tree,chain,general}.csv
  ↓
analyze_mixed_eval.py
  ↓
Outputs:

1. TV vs Context Length (all structures)
   - Line plot showing learning curves
   - Decreasing TV = good learning
   
2. TV Heatmap
   - Grid: (structures × context_lengths)
   - Color: green (low TV) to red (high TV)
   
3. Per-Target Analysis
   - Separate curves for each node (A, B, C, D, E)
   - Some nodes may be easier to predict
   
4. Summary Statistics
   - Mean/std TV per structure
   - Performance at different context lengths
```

## Component Details

### 1. Graph Structures (`data/mixed_graphs.py`)

**Design Principles:**
- All 3 structures have **exactly 5 nodes** (A, B, C, D, E)
- Nodes are binary: {0, 1}
- Each structure represents a different DAG topology
- CPTs are randomly sampled for each graph instance

**Functions:**
- `get_tree_5node(seed)` - Binary tree with 4 edges
- `get_chain_5node(seed)` - Linear chain with 4 edges
- `get_general_5node(seed)` - Complex DAG with 6 edges
- `get_mixed_graph_structures(seed)` - Returns all 3

### 2. Dataset (`data/mixed_dataset.py`)

**Key Class: `MixedGraphICLSequenceDataset`**

**Initialization:**
```python
dataset = MixedGraphICLSequenceDataset(
    templates=[tree, chain, general],    # 3 compiled templates
    p1_lists=[p1_tree, p1_chain, p1_gen], # CPTs for each
    spec=MixedICLBatchSpec(
        batch_graphs=12,      # Total graphs per batch
        num_example=100,      # Context length
        target_index=random,  # Which node to predict
    )
)
```

**Batch Generation Logic:**
1. Split `batch_graphs` evenly across structures (4+4+4 for batch=12)
2. For each structure:
   - Sample graph IDs (which CPT to use)
   - Generate L observations from those graphs
   - Apply causal masking (future nodes masked)
3. Concatenate and optionally shuffle
4. Return batch with structure IDs

**Output Shape:** `(B, L, N+1)` where:
- B = batch_graphs (12)
- L = num_example + 1 (101)
- N+1 = 5 nodes + 1 target index (6)

### 3. Training (`train_mixed.py`)

**Key Arguments:**
- `--context-len`: Context examples per sample (default: 100)
- `--train-size`: Graph instances per structure (default: 20000)
- `--train-step`: Optimization steps (default: 50000)
- `--batch-size`: Total graphs per batch (default: 64)
- `--seed`: Random seed for reproducibility

**Training Loop:**
1. Load 3 graph structures
2. Sample CPTs for each structure (train_size graphs)
3. Create MixedGraphICLSequenceDataset
4. Train using PyTorch Lightning
5. Save best checkpoint based on validation loss

### 4. Evaluation (`eval_mixed.py`)

**Key Innovation:** Separate per-structure evaluation

**Process:**
1. Load trained checkpoint (one model)
2. For each structure:
   - Create **one fixed graph** (fixed CPT)
   - Evaluate across context lengths [1, 2, 5, ..., 500]
   - Run 1000 episodes per context length
   - Compute TV: |p_predicted - p_true|
   - Save to CSV

**Why This Matters:**
- Tests if model learns **structure** from context
- Tests if model learns **CPT parameters** from context
- Different structures may have different difficulty

### 5. Analysis (`analyze_mixed_eval.py`)

**Generated Plots:**

**a) TV vs Context (all structures):**
- X-axis: Context length (log scale)
- Y-axis: Mean TV
- One line per structure
- Shows learning curves

**b) Per-Target Analysis:**
- One subplot per structure
- Multiple lines (one per target node)
- Reveals which nodes are harder to predict

**c) Heatmap:**
- Rows: Structures
- Columns: Context lengths
- Color: TV value (green=good, red=bad)

**Summary Statistics:**
- Overall TV (mean ± std)
- Per context length breakdown
- Per target node breakdown
- Cross-structure comparison

## Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                         TRAINING PHASE                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Graph Structures ──┬──> Sample CPTs ──┬──> Dataset ──> Batches   │
│  (3 templates)      │   (20k per type)  │   (balanced)   (mixed)   │
│                     │                   │                           │
│  • Tree             │   • tree_cpts     │   Each batch:             │
│  • Chain            │   • chain_cpts    │   [4 tree,                │
│  • General          │   • gen_cpts      │    4 chain,               │
│                     │                   │    4 general]             │
│                     │                   │                           │
│                     └──> Context Mask ──┘   (L, N+1) per sample    │
│                         (future masked)                             │
│                                                                     │
│                              ↓                                      │
│                                                                     │
│                        Transformer Model                            │
│                     (learns from all 3)                             │
│                              ↓                                      │
│                                                                     │
│                        Save Checkpoint                              │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                        EVALUATION PHASE                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Checkpoint ───┬──> Eval Tree ───> eval_tv_tree.csv               │
│  (one model)   ├──> Eval Chain ──> eval_tv_chain.csv              │
│                └──> Eval General -> eval_tv_general.csv            │
│                                                                     │
│  Each evaluation:                                                   │
│  • 1 fixed graph (fixed CPT)                                       │
│  • Context lengths: [1, 2, 5, 10, ..., 500]                        │
│  • 1000 episodes each                                               │
│  • Measure TV = |p_hat - p_true|                                   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                         ANALYSIS PHASE                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  CSV Files ──> analyze_mixed_eval.py ──> Plots + Statistics        │
│                                                                     │
│  Outputs:                                                           │
│  • tv_vs_context_all_structures.png                                │
│  • tv_per_target.png                                               │
│  • tv_heatmap.png                                                  │
│  • summary_statistics.txt                                          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Key Design Decisions

### 1. Why 5 Nodes?
- Small enough for tractable computation
- Large enough for interesting structure
- Allows comparison with existing work

### 2. Why These 3 Structures?
- **Tree**: Hierarchical dependencies
- **Chain**: Sequential dependencies  
- **General**: Complex multi-parent dependencies
- Cover different topology types

### 3. Why Balanced Sampling?
- Ensures no structure is over/under-represented
- Fair training signal for all structures
- Prevents model bias toward one type

### 4. Why Separate Evaluation?
- Tests structure learning specifically
- Reveals which structures are harder
- Allows targeted analysis per structure

### 5. Why Fixed Graph in Evaluation?
- Isolates in-context learning ability
- Consistent comparison across context lengths
- Removes CPT variability from results

## Performance Interpretation

### Good Model Behavior
```
Context | Tree | Chain | General
--------|------|-------|--------
1       | 0.30 | 0.30  | 0.30    ← All start similar (no info)
10      | 0.15 | 0.20  | 0.25    ← Learning begins
100     | 0.05 | 0.08  | 0.12    ← Strong learning
500     | 0.02 | 0.04  | 0.08    ← Convergence
```
✓ TV decreases with context for all structures
✓ Final TV < 0.10 for all
✓ General may be harder (more edges)

### Poor Model Behavior
```
Context | Tree | Chain | General
--------|------|-------|--------
1       | 0.30 | 0.30  | 0.30
10      | 0.28 | 0.29  | 0.30    ← No improvement
100     | 0.25 | 0.27  | 0.31    ← Minimal learning
500     | 0.24 | 0.26  | 0.32    ← Plateaued
```
✗ TV doesn't decrease much
✗ Final TV > 0.20
✗ Model not learning from context

## Extension Points

### Add More Structures
Edit `data/mixed_graphs.py`:
```python
def get_star_5node(seed=2000):
    # Star topology: A in center, B-E around it
    ...

def get_mixed_graph_structures(seed=2000):
    return [
        get_tree_5node(seed),
        get_chain_5node(seed),
        get_general_5node(seed),
        get_star_5node(seed + 3),  # Add new structure
    ]
```

### Change Node Count
- Modify each `get_*_5node()` to use N nodes
- Update CPT sampling (`random_binary_cpt(num_parents)`)
- Adjust dataset `num_nodes` parameter

### Different Evaluation Metrics
Edit `utils/evaluate.py`:
- Add KL divergence
- Add accuracy metrics
- Add structure recovery metrics

## Troubleshooting

**Issue:** Batch distribution uneven
- Check: `samples_per_structure` in dataset
- Fix: Adjust batch_size to be divisible by num_structures

**Issue:** TV not decreasing
- Check: Model capacity (layers, heads)
- Check: Learning rate
- Check: Training steps (may need more)

**Issue:** One structure performs poorly
- Check: Graph structure definition (valid DAG?)
- Check: CPT sampling (reasonable probabilities?)
- Check: Evaluation uses correct structure

## References

For implementation details, see:
- [USAGE_GUIDE.md](USAGE_GUIDE.md) - How to run experiments
- [DEVELOPMENT.md](DEVELOPMENT.md) - Development checklist
- Source code comments in each file
