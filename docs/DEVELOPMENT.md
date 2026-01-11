# Development Guide

Notes for developers working on the Mixed Graph Structure Training System.

## Project Status

### ✅ Completed Features

**Core Implementation:**
- [x] 3 graph structures (tree, chain, general) with 5 nodes each
- [x] Balanced dataset for multi-structure training
- [x] Training script compatible with existing codebase
- [x] Per-structure evaluation strategy
- [x] Analysis and visualization tools
- [x] Comprehensive unit tests
- [x] SLURM job scripts for cluster
- [x] Complete documentation

**Key Files:**
- `data/mixed_graphs.py` - Graph structure definitions
- `data/mixed_dataset.py` - Balanced dataset implementation
- `train_mixed.py` - Training script
- `eval_mixed.py` - Evaluation script
- `analyze_mixed_eval.py` - Analysis script
- `test_mixed_structures.py` - Unit tests

### 🔄 In Progress

- [ ] Training runs across multiple seeds
- [ ] Comprehensive evaluation results
- [ ] Cross-model comparisons

### 📋 Future Work

- [ ] Add more graph structures (star, fully-connected, etc.)
- [ ] Support variable node counts
- [ ] Add structure recovery metrics
- [ ] Implement attention visualization
- [ ] Add early stopping based on validation TV

## Development Workflow

### 1. Setup Development Environment

```bash
# Clone repository
git clone <repository-url>
cd JASACODE

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
uv sync

# Or with pip
pip install -r requirements.txt
```

### 2. Make Changes

**Adding a new graph structure:**
1. Edit `data/mixed_graphs.py`
2. Add function `get_newstructure_5node(seed)`
3. Update `get_mixed_graph_structures()`
4. Add test in `test_mixed_structures.py`
5. Run tests: `python test_mixed_structures.py`

**Modifying dataset:**
1. Edit `data/mixed_dataset.py`
2. Update `MixedGraphICLSequenceDataset` class
3. Add/update tests
4. Verify batch distribution is balanced

**Changing evaluation:**
1. Edit `eval_mixed.py`
2. Modify evaluation loop or metrics
3. Ensure CSV output format consistent
4. Update analysis script if needed

### 3. Test Changes

```bash
# Run unit tests
python test_mixed_structures.py

# Quick training test
python train_mixed.py \
  --context-len 10 \
  --train-size 100 \
  --train-step 500 \
  --batch-size 12 \
  --output-dir test_run

# Quick evaluation test
python eval_mixed.py \
  --checkpoint test_run/.../best.ckpt \
  --output-dir test_eval \
  --test-size 50 \
  --context-lens 1 10

# Verify analysis works
python analyze_mixed_eval.py --eval-dir test_eval
```

### 4. Commit and Push

```bash
git add <files>
git commit -m "Description of changes"
git push origin <your-branch>
```

## Code Structure

### Data Module (`data/`)

**`mixed_graphs.py`:**
- Purpose: Define graph structures
- Key functions:
  - `get_tree_5node(seed)`: Binary tree
  - `get_chain_5node(seed)`: Linear chain
  - `get_general_5node(seed)`: Complex DAG
  - `get_mixed_graph_structures(seed)`: Returns all
- Dependencies: `binary_bn.BinaryBayesNet`

**`mixed_dataset.py`:**
- Purpose: Multi-structure dataset
- Key classes:
  - `MixedICLBatchSpec`: Batch configuration
  - `MixedGraphICLSequenceDataset`: Dataset implementation
- Key methods:
  - `__init__()`: Setup templates and CPTs
  - `__len__()`: Return number of batches
  - `__getitem__()`: Generate balanced batch
- Dependencies: `multigraph_sampler.sample_many_graphs`

**`__init__.py`:**
- Exports all public functions/classes
- Maintains backward compatibility

### Training Module

**`train_mixed.py`:**
- Purpose: Train model on mixed structures
- Key functions:
  - `get_args()`: Parse command-line arguments
  - `main()`: Setup and run training
- Uses: PyTorch Lightning for training loop
- Creates: Checkpoints, logs, config files

### Evaluation Module

**`eval_mixed.py`:**
- Purpose: Evaluate on each structure separately
- Key functions:
  - `get_args()`: Parse arguments
  - `main()`: Run evaluation for all structures
- Key insight: One checkpoint → multiple evaluations
- Output: CSV per structure with TV metrics

### Analysis Module

**`analyze_mixed_eval.py`:**
- Purpose: Visualize and analyze results
- Key functions:
  - `load_eval_results()`: Load CSVs
  - `plot_tv_vs_context()`: Main comparison plot
  - `plot_per_target_tv()`: Per-node analysis
  - `plot_tv_heatmap()`: Context × structure grid
  - `generate_summary_stats()`: Text statistics
- Dependencies: matplotlib, seaborn, pandas

### Testing Module

**`test_mixed_structures.py`:**
- Purpose: Unit tests for all components
- Tests:
  - Graph structure creation
  - Dataset batch generation
  - Masking correctness
  - Structure ID tracking
  - Balanced distribution
- Run: `python test_mixed_structures.py`

## Adding New Features

### Example 1: Add Star Graph Structure

**Step 1: Define structure in `data/mixed_graphs.py`:**
```python
def get_star_5node(seed=2000):
    """
    Star structure: A in center, B-E as leaves
    
    Structure:
        B   C
         \ /
          A
         / \
        D   E
    """
    rng = np.random.default_rng(seed)
    bn = BinaryBayesNet()
    
    # Add nodes
    for n in ["A", "B", "C", "D", "E"]:
        bn.add_node(n)
    
    # Star edges (all point to A)
    bn.add_edge("A", "B")
    bn.add_edge("A", "C")
    bn.add_edge("A", "D")
    bn.add_edge("A", "E")
    
    # Set parents
    bn.set_parents("A", [])  # A is root
    bn.set_parents("B", ["A"])
    bn.set_parents("C", ["A"])
    bn.set_parents("D", ["A"])
    bn.set_parents("E", ["A"])
    
    # Random CPTs
    bn.set_cpt("A", random_binary_cpt(0, rng))
    for node in ["B", "C", "D", "E"]:
        bn.set_cpt(node, random_binary_cpt(1, rng))
    
    return bn
```

**Step 2: Update structure list:**
```python
def get_mixed_graph_structures(seed=2000):
    return [
        get_tree_5node(seed),
        get_chain_5node(seed),
        get_general_5node(seed),
        get_star_5node(seed + 3),  # Add new structure
    ]

def get_structure_names():
    return ["tree", "chain", "general", "star"]
```

**Step 3: Add test:**
```python
def test_star_structure():
    star = get_star_5node(seed=2000)
    assert star.num_nodes == 5
    assert len(star.edges()) == 4  # 4 edges from A
    # All nodes except A have A as parent
    for node in ["B", "C", "D", "E"]:
        assert star.parents(node) == ["A"]
```

**Step 4: Run tests and train:**
```bash
python test_mixed_structures.py
python train_mixed.py --output-dir runs/with_star
```

### Example 2: Add KL Divergence Metric

**Step 1: Add to `utils/evaluate.py` (or create new file):**
```python
def compute_kl_divergence(p_hat, p_true, epsilon=1e-8):
    """
    Compute KL divergence: KL(p_true || p_hat)
    
    Args:
        p_hat: Predicted probability
        p_true: True probability
        epsilon: Small value to avoid log(0)
    
    Returns:
        KL divergence value
    """
    p_hat = np.clip(p_hat, epsilon, 1 - epsilon)
    p_true = np.clip(p_true, epsilon, 1 - epsilon)
    
    kl = (p_true * np.log(p_true / p_hat) + 
          (1 - p_true) * np.log((1 - p_true) / (1 - p_hat)))
    
    return kl
```

**Step 2: Update `eval_mixed.py` to compute KL:**
```python
# In the evaluation loop, after computing TV
kl = compute_kl_divergence(p_hat, p_true)

# Add to results
results.append({
    "context_len": context_len,
    "target_index": target_index,
    "episode": ep,
    "p_hat": p_hat,
    "p_true": p_true,
    "tv": tv,
    "kl": kl,  # Add KL
    "y_test": y_test,
    "parents_cfg": parents_cfg,
})
```

**Step 3: Update analysis to plot KL:**
```python
def plot_kl_vs_context(results, output_dir):
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    for name, df in results.items():
        summary = df.groupby("context_len")["kl"].mean()
        ax.plot(summary.index, summary.values, 
                marker='o', label=name)
    
    ax.set_xlabel("Context Length")
    ax.set_ylabel("Mean KL Divergence")
    ax.legend()
    plt.savefig(output_dir / "kl_vs_context.png")
```

### Example 3: Variable Node Count

**Step 1: Parameterize graph functions:**
```python
def get_tree_variable(num_nodes, seed=2000):
    """
    Create tree with variable number of nodes.
    """
    assert num_nodes >= 2, "Need at least 2 nodes"
    
    bn = BinaryBayesNet()
    rng = np.random.default_rng(seed)
    
    # Create nodes
    node_names = [chr(65 + i) for i in range(num_nodes)]  # A, B, C, ...
    for name in node_names:
        bn.add_node(name)
    
    # Build binary tree structure
    for i in range(1, num_nodes):
        parent_idx = (i - 1) // 2
        parent = node_names[parent_idx]
        child = node_names[i]
        bn.add_edge(parent, child)
    
    # Set parents and CPTs
    # ... implementation
    
    return bn
```

**Step 2: Update dataset to support variable N:**
```python
class MixedGraphICLSequenceDataset:
    def __init__(self, templates, p1_lists, spec, ...):
        # Allow different num_nodes per structure
        self.num_nodes_per_structure = [t.num_nodes for t in templates]
        
        # Validate all have same num_nodes (for now)
        assert len(set(self.num_nodes_per_structure)) == 1, \
            "All structures must have same num_nodes (for now)"
```

## Testing Guidelines

### Unit Tests

**What to test:**
- Graph structure creation (valid DAG, correct edges)
- CPT sampling (valid probabilities)
- Dataset batch generation (correct shape, balanced)
- Masking logic (future nodes masked correctly)
- Structure ID tracking (correct labels)

**Test template:**
```python
def test_feature_name():
    """Test description."""
    # Setup
    input_data = ...
    
    # Execute
    result = function_under_test(input_data)
    
    # Verify
    assert result.shape == expected_shape
    assert result.dtype == expected_dtype
    assert np.all(result >= 0)  # Example constraint
```

### Integration Tests

**Quick smoke test:**
```bash
# Train for 100 steps
python train_mixed.py \
  --train-step 100 \
  --train-size 10 \
  --batch-size 12 \
  --context-len 5 \
  --output-dir smoke_test

# Should complete without errors
# Check checkpoint exists
ls smoke_test/.../checkpoints/last.ckpt
```

### Manual Testing Checklist

Before committing changes:

- [ ] Unit tests pass
- [ ] Code follows existing style
- [ ] Docstrings updated
- [ ] No hardcoded paths
- [ ] No debug print statements
- [ ] Training runs without errors
- [ ] Evaluation runs without errors
- [ ] Analysis generates plots
- [ ] Documentation updated

## Performance Optimization

### Training Speed

**Current bottlenecks:**
1. Data generation (sampling from BN)
2. Masking operations
3. Forward pass through transformer

**Optimization strategies:**
1. **Pre-generate data:**
   ```python
   # Generate all training data upfront
   all_data = [dataset[i] for i in range(len(dataset))]
   ```

2. **Use DataLoader with multiple workers:**
   ```python
   dataloader = DataLoader(
       dataset,
       batch_size=1,  # Dataset already returns batches
       num_workers=4,
       pin_memory=True
   )
   ```

3. **Profile code:**
   ```bash
   python -m cProfile -o profile.stats train_mixed.py
   python -m pstats profile.stats
   ```

### Memory Optimization

**If OOM (Out of Memory):**
1. Reduce batch_size
2. Reduce context_len
3. Use gradient accumulation
4. Reduce embedding_dim or num_layer

**Example gradient accumulation:**
```python
# In trainer
trainer = pl.Trainer(
    accumulate_grad_batches=4,  # Effective batch_size × 4
    ...
)
```

## Code Style

### Python Style Guide

Follow PEP 8:
- 4 spaces for indentation
- Max line length: 88 characters (Black formatter)
- Docstrings for all public functions/classes

**Function docstring template:**
```python
def function_name(arg1, arg2, optional_arg=None):
    """
    Brief description of what function does.
    
    Args:
        arg1: Description of arg1
        arg2: Description of arg2
        optional_arg: Description of optional_arg (default: None)
    
    Returns:
        Description of return value
        
    Raises:
        ValueError: When invalid input provided
    """
    pass
```

### Git Commit Messages

**Format:**
```
Short summary (50 chars or less)

Longer explanation if needed. Wrap at 72 characters.
- Bullet points okay
- Multiple paragraphs okay

Fixes #123
```

**Examples:**
```
Add star graph structure

Implements star topology where central node A
connects to all leaf nodes B, C, D, E.

Add KL divergence metric to evaluation

Computes KL(p_true || p_hat) in addition to TV.
Updates analysis to plot KL curves.

Fix batch distribution for non-divisible batch sizes

Previously crashed when batch_size not divisible by 3.
Now handles remainder correctly.
```

## Debugging Tips

### Common Issues

**Issue: "AssertionError: All structures must have 5 nodes"**
- Check: Graph definition has exactly 5 `add_node()` calls
- Check: No duplicate nodes

**Issue: "Batch distribution not balanced"**
- Check: batch_size divisible by num_structures
- Or: Update logic to handle remainder

**Issue: "TV not decreasing"**
- Check: Training loss decreasing
- Check: Model has enough capacity
- Check: Learning rate not too high/low
- Check: Masking applied correctly

### Debug Mode

**Add debug prints:**
```python
# In dataset
def __getitem__(self, idx):
    print(f"Generating batch {idx}")
    print(f"Samples per structure: {samples_per_structure}")
    ...
```

**Visualize batch:**
```python
# After getting batch
batch = dataset[0]
X = batch["X"]  # (B, L, N+1)
print(f"Batch shape: {X.shape}")
print(f"Structure IDs: {batch['struct_id']}")
print(f"Sample from batch:\n{X[0, :5, :]}")  # First 5 rows
```

**Check gradients:**
```python
# After backward pass
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad norm = {param.grad.norm().item():.4f}")
```

## Documentation

### Where to Document

**Code comments:**
- Complex algorithms
- Non-obvious design decisions
- Performance considerations

**Docstrings:**
- All public functions/classes
- Module-level docstring in each file

**Markdown files:**
- High-level architecture (ARCHITECTURE.md)
- Usage instructions (USAGE_GUIDE.md)
- Development notes (this file)

### Updating Documentation

When adding features:
1. Update relevant .md files
2. Add docstrings to new functions
3. Update README.md if user-facing
4. Add examples if helpful

## Release Checklist

Before creating a release:

- [ ] All tests pass
- [ ] Code reviewed
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Version number bumped
- [ ] Example runs completed
- [ ] README.md accurate
- [ ] No TODO/FIXME in critical code

## Contact

For development questions:
- Open GitHub issue
- Check existing documentation
- Review code comments

## Appendix: File Reference

```
JASACODE/
├── data/
│   ├── __init__.py              # Exports
│   ├── mixed_graphs.py          # Graph structures (218 lines)
│   ├── mixed_dataset.py         # Dataset (200 lines)
│   ├── binary_bn.py             # BayesNet class (existing)
│   └── multigraph_sampler.py    # Sampling utils (existing)
├── model/
│   └── icl_model.py             # Transformer model (existing)
├── utils/
│   ├── evaluate.py              # Evaluation utils (existing)
│   └── ...
├── train_mixed.py               # Training script (218 lines)
├── eval_mixed.py                # Evaluation script (181 lines)
├── analyze_mixed_eval.py        # Analysis script (258 lines)
├── test_mixed_structures.py     # Unit tests (195 lines)
├── run_mixed_pipeline.sh        # Pipeline script
├── train_mixed_job.sh           # SLURM training
├── eval_mixed_job.sh            # SLURM evaluation
├── docs/
│   ├── README.md                # Main documentation
│   ├── ARCHITECTURE.md          # System design
│   ├── USAGE_GUIDE.md           # Usage instructions
│   └── DEVELOPMENT.md           # This file
└── pyproject.toml               # Dependencies
```

**Total new code:** ~1500 lines
**Documentation:** ~4000 lines
