# Collaborator Changes Summary

**Date:** January 10, 2026  
**Commits:** 3 new commits on `origin/main`
- `6e646b4` - "ok" (10 hours ago)
- `9574a0e` - "tv regress" (24 hours ago)  
- `d28e8a1` - "baselines" (earlier)

**Total Changes:** 786 insertions, 100 deletions across 11 files

---

## 🔑 Major Changes Overview

### 1. **Dynamic Context Length** (`data/dataset.py`)
**Impact: HIGH** - Core dataset functionality changed

**What Changed:**
- `ICLBatchSpec` now supports **dynamic context lengths**
- Added fields: `min_context_len`, `max_context_len`
- Made `num_example` optional (use either fixed or dynamic)

**Before:**
```python
@dataclass
class ICLBatchSpec:
    batch_graphs: int
    num_example: int  # Fixed context length
    target_index: int
```

**After:**
```python
@dataclass
class ICLBatchSpec:
    batch_graphs: int
    target_index: int
    num_example: Optional[int] = None              # Fixed context (if specified)
    min_context_len: Optional[int] = None          # Dynamic min
    max_context_len: Optional[int] = None          # Dynamic max
```

**How it works:**
- If `num_example` is set → fixed context length (like before)
- If `min/max_context_len` are set → randomly sample context length per batch
- Must specify one or the other

---

### 2. **Target Label Changed to Probability** (`data/dataset.py`)
**Impact: CRITICAL** - Changes loss function and training objective

**What Changed:**
- `batch["y"]` is now a **probability estimate from context**, not binary label
- Uses Bayesian CPT estimation with known DAG structure
- Estimates `P(X_t=1 | parent_config)` from context examples

**Before:**
```python
y = X_full[:, L - 1, t].astype(np.int64)  # (B,) - binary label {0, 1}
```

**After:**
```python
# Estimate CPT from context using Bayesian inference
y = np.empty((B,), dtype=np.float32)  # (B,) - probabilities [0.0, 1.0]

# For each batch element:
# 1. Count parent config occurrences in context
# 2. Estimate P(X_t=1 | cfg) = (alpha + count_1) / (alpha + beta + count_total)
# 3. Use Beta prior (alpha=beta=0.0 = maximum likelihood)
```

**Algorithm:**
```python
for i in range(B):
    # Count occurrences of each parent configuration
    tot = bincount(cfg_ctx[i])  # Total times each config appears
    one = bincount(cfg_ctx[i], weights=y_ctx[i])  # Times it equals 1
    cfg = cfg_test[i]  # Parent config of test example
    
    if tot[cfg] > 0:
        y[i] = (alpha + one[cfg]) / (alpha + beta + tot[cfg])
    else:
        # Fall back to marginal if config not seen in context
        y[i] = mean(y_ctx[i])
```

**Implications:**
- Training now predicts **probabilities**, not hard labels
- More nuanced learning signal
- Matches evaluation metric (Total Variation distance)

---

### 3. **Loss Function Changed** (`utils/trainer.py`)
**Impact: CRITICAL** - Training objective fundamentally different

**Before:**
```python
loss = F.binary_cross_entropy_with_logits(logits, y.float())
```

**After:**
```python
p_hat = torch.sigmoid(logits)
loss = F.mse_loss(p_hat, y.float())  # MSE instead of BCE
```

**Why:**
- `y` is now probability, not binary label
- MSE measures distance between predicted and estimated probabilities
- Aligns with evaluation metric (Total Variation)

**Metric Changed:**
```python
# Before: Accuracy
pred = (torch.sigmoid(logits) > 0.5).long()
acc = (pred == y).float().mean()

# After: Total Variation
p_hat = torch.sigmoid(logits)
tv = torch.abs(p_hat - y.float()).mean()
```

---

### 4. **Learning Rate Scheduler** (`utils/trainer.py`)
**Impact: MEDIUM** - Training dynamics improved

**Added:**
- Linear warmup for first 1000 steps
- Cosine annealing decay after warmup
- Minimum LR floor

**New Parameters:**
```python
class ICLLightningModule:
    def __init__(
        self,
        max_steps: int = 100_000,      # NEW
        warmup_steps: int = 1000,       # NEW
        min_lr: float = 0.0,            # NEW
        ...
    )
```

**Scheduler Logic:**
```python
def configure_optimizers(self):
    optimizer = torch.optim.AdamW(...)
    
    # 1. Warmup: linear increase from 0 to init_lr
    warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, 
                      total_iters=warmup_steps)
    
    # 2. Cosine decay: from init_lr to min_lr
    cosine = CosineAnnealingLR(optimizer, T_max=max_steps - warmup_steps, 
                               eta_min=min_lr)
    
    # Combine sequentially
    scheduler = SequentialLR(optimizer, [warmup, cosine], [warmup_steps])
```

---

### 5. **Baseline Evaluation Methods** (`utils/evaluate.py`)
**Impact: MEDIUM** - New evaluation baselines added

**Two New Baselines:**

#### a) Naive Marginal Baseline
```python
def _naive_marginal_from_context(X_full, target_index, alpha=0.0, beta=0.0):
    """
    Simplest baseline: estimate P(X_t=1) from context marginal only.
    Ignores graph structure and parents completely.
    """
    m = L - 1  # context length
    ctx = X_full[:, :m, t]  # context examples
    return (alpha + ctx.sum()) / (alpha + beta + m)
```

#### b) Bayesian CPT Baseline
```python
def _bayes_cpt_from_context(X_full, template, target_index, alpha=0.0, beta=0.0):
    """
    Bayesian baseline with KNOWN DAG structure:
    - Estimates CPT from context for each parent configuration
    - Uses structure information (parent_idx)
    - Returns P(X_t=1 | parent_config_test)
    """
    # Same algorithm as training label generation
    # Provides upper bound for in-context learning
```

**Purpose:**
- Compare model performance against statistical baselines
- Naive baseline = lower bound (ignores structure)
- Bayesian baseline = upper bound (uses known structure)

---

### 6. **Model Architecture Change** (`model/models.py`)
**Impact: LOW** - Minor tweak

**Added dropout to output layer:**
```python
class BinaryHead(nn.Module):
    def __init__(self, embed_dim: int, dropout: float = 0.1):  # NEW dropout param
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 1),
            nn.Dropout(dropout),  # NEW: regularization
        )
```

---

### 7. **Enhanced Training Script** (`train.py`)
**Impact: HIGH** - Many new features

**New Arguments:**
```python
# Dynamic context
--min-context-len    # Dynamic context min (e.g., 10)
--max-context-len    # Dynamic context max (e.g., 200)

# Scheduler
--warmup-steps       # LR warmup steps (default: 1000)
--min-lr             # Minimum LR (default: 0.0)

# Evaluation
--eval-interval      # Steps between evaluations
--eval-context-lens  # List of context lengths for eval
--test-size          # Test episodes per context length
```

**New Features:**
- Support for dynamic context length training
- Periodic evaluation during training
- Baseline comparison (naive + Bayesian)
- Enhanced logging

---

### 8. **Visualization Improvements** (`runs/quick_plot.py`)
**Impact: LOW** - Better plotting

**Added:**
- TV vs context plots with baselines
- Error bars / confidence intervals
- Multiple experiment comparison
- Automatic baseline plotting

---

### 9. **Job Script Updates** (`train_job.sh`)
**Impact: MEDIUM** - New SLURM job patterns

**Changes:**
- Grid search over hyperparameters
- Multiple seeds per configuration
- Dynamic context length experiments
- Better resource allocation

---

## 🚨 Breaking Changes for Mixed Graph Code

### Must Update:

#### 1. **MixedICLBatchSpec** → Add dynamic context support
```python
@dataclass
class MixedICLBatchSpec:
    batch_graphs: int
    target_index: int
    num_example: Optional[int] = None              # Make optional
    min_context_len: Optional[int] = None          # Add these
    max_context_len: Optional[int] = None          # Add these
    dtype: torch.dtype = torch.long
    device: Optional[torch.device] = None
```

#### 2. **Change Label Generation** in `MixedGraphICLSequenceDataset`
- Current: Returns binary label from test row
- New: Must compute Bayesian CPT estimate from context
- **This is complex** - need to handle 3 different structures

#### 3. **Update Training Loss** in `train_mixed.py`
```python
# Change from:
loss = F.binary_cross_entropy_with_logits(logits, y.float())

# To:
p_hat = torch.sigmoid(logits)
loss = F.mse_loss(p_hat, y.float())
```

#### 4. **Update Logging Metrics**
```python
# Change from accuracy to TV
with torch.no_grad():
    tv = torch.abs(p_hat - y.float()).mean()
self.log("train/tv", tv, ...)
```

#### 5. **Add LR Scheduler Parameters**
```python
# In ICLLightningModule initialization
module = ICLLightningModule(
    ...,
    max_steps=args.train_step,
    warmup_steps=1000,
    min_lr=0.0,
)
```

---

## 📊 Expected Impact on Mixed Training

### Positive:
✅ More sophisticated training objective (probabilities vs labels)
✅ Better LR scheduling → faster convergence
✅ Evaluation baselines → better performance understanding
✅ Dynamic context → more robust model

### Challenges:
⚠️ Label generation more complex (3 structures × Bayesian estimation)
⚠️ Need to test if MSE loss works well for mixed structures
⚠️ Dynamic context may need tuning for mixed case

---

## 🔧 Migration Strategy

### Phase 1: Minimal Changes (Get It Working)
1. Keep fixed context length (`num_example`)
2. Update loss function to MSE
3. Add LR scheduler parameters
4. Test that training works

### Phase 2: Add Bayesian Labels
1. Implement label estimation in `MixedGraphICLSequenceDataset`
2. Verify correctness with unit tests
3. Compare training curves (old vs new)

### Phase 3: Full Feature Parity
1. Add dynamic context length support
2. Add baseline evaluation functions
3. Update job scripts with new arguments

---

## 📝 Action Items

### High Priority (Required):
- [ ] Update `MixedICLBatchSpec` to match new `ICLBatchSpec`
- [ ] Change loss function from BCE to MSE
- [ ] Add LR scheduler parameters
- [ ] Update logging metrics (acc → tv)

### Medium Priority (Recommended):
- [ ] Implement Bayesian label generation
- [ ] Add baseline evaluation functions
- [ ] Test dynamic context length

### Low Priority (Nice to Have):
- [ ] Update visualization scripts
- [ ] Add grid search to job scripts
- [ ] Enhance documentation

---

## 🧪 Testing Plan

After applying changes:

1. **Sanity Check:**
   ```bash
   python test_mixed_structures.py  # Unit tests still pass
   ```

2. **Quick Training Test:**
   ```bash
   python train_mixed.py --train-step 1000 --batch-size 12
   # Should complete without errors
   ```

3. **Verify Loss:**
   - Loss should be MSE (range ~0.0-1.0, not BCE range)
   - TV metric should be logged

4. **Compare Results:**
   - Train with old code → save results
   - Train with new code → compare TV curves
   - New should be similar or better

---

## 📚 Files to Review in Detail

**Critical:**
1. `data/dataset.py` - Lines 70-168 (label generation)
2. `utils/trainer.py` - Lines 26-94 (loss & scheduler)
3. `utils/evaluate.py` - Lines 74-285 (baselines)

**Important:**
4. `train.py` - New arguments and evaluation logic
5. `train_job.sh` - New hyperparameter patterns

---

## 💡 Key Insights

1. **Philosophy Change:** Model now learns to estimate **probabilities** from context, not just predict labels
2. **More Realistic:** Target labels come from context (like real ICL), not ground truth
3. **Better Evaluation:** MSE loss matches TV evaluation metric
4. **Scalability:** Dynamic context prepares for curriculum learning

---

## Questions to Ask Collaborator

1. Why switch from BCE to MSE loss?
   - Answer: Aligns with probability estimation objective
   
2. Should mixed structures also use Bayesian labels?
   - Likely: Yes, for consistency
   
3. What are recommended hyperparameters for new scheduler?
   - Default: warmup_steps=1000, likely fine
   
4. Any issues with dynamic context you encountered?
   - May have insights on good min/max ranges

---

## Next Steps

1. Review this document thoroughly
2. Decide on migration strategy (Phase 1 vs full)
3. I'll help implement the changes step by step
4. Test thoroughly before pushing

---

**Status:** Ready for your review and decision on migration approach.
