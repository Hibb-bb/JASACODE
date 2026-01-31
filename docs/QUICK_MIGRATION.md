# Quick Reference: Applying Collaborator Changes

## 🎯 TL;DR - What Changed

Your collaborator made **9 major changes** to single-graph training code:

1. ⚠️ **CRITICAL:** Target labels are now **probabilities** (not binary)
2. ⚠️ **CRITICAL:** Loss changed from **BCE → MSE**
3. ✅ **GOOD:** Added **LR scheduler** (warmup + cosine decay)
4. ✅ **GOOD:** Added **dynamic context length** support
5. ✅ **GOOD:** Added **baseline evaluation** methods
6. 📊 **MINOR:** Better logging, plotting, job scripts

**Impact on your code:** Need to update 3 core files in mixed graph system.

---

## 🚀 Quick Migration Path

### Option A: Minimal Update (30 minutes)
**Goal:** Get mixed training working with new changes

**Steps:**
1. Update loss function (1 line change)
2. Add scheduler parameters (3 lines)
3. Test training runs

**Risk:** Low, backward compatible

### Option B: Full Update (2-3 hours)
**Goal:** Feature parity with single-graph code

**Steps:**
1. Do Option A
2. Implement Bayesian label generation
3. Add dynamic context support
4. Add baseline evaluations

**Risk:** Medium, more testing needed

---

## 📝 File-by-File Changes Needed

### File 1: `data/mixed_dataset.py` (Core dataset)

**Location:** Line ~98-100 in `__getitem__` method

**Current Code:**
```python
y = X_full[:, L - 1, t].astype(np.int64)  # Binary label
```

**New Code:** (Option A - Simple)
```python
# TODO: For now, keep binary labels but change dtype
y = X_full[:, L - 1, t].astype(np.float32)  # Will be 0.0 or 1.0
```

**New Code:** (Option B - Full Bayesian)
```python
# Compute Bayesian CPT estimate from context (per structure)
# Copy logic from data/dataset.py lines 109-154
# Need to handle: each batch element may have different structure_id
# See COLLABORATOR_CHANGES.md for full algorithm
```

---

### File 2: `train_mixed.py` (Training script)

**Location:** Training loop (ICLLightningModule creation)

**Current Code:**
```python
module = ICLLightningModule(
    input_dim=6,
    init_lr=args.lr,
    weight_decay=args.weight_decay,
    # ... model params
)
```

**New Code:**
```python
module = ICLLightningModule(
    input_dim=6,
    init_lr=args.lr,
    weight_decay=args.weight_decay,
    max_steps=args.train_step,      # NEW
    warmup_steps=1000,               # NEW
    min_lr=0.0,                      # NEW
    # ... model params
)
```

**Note:** `utils/trainer.py` already has the new loss function (MSE), so this will automatically apply!

---

### File 3: `data/mixed_dataset.py` (Batch spec)

**Location:** Class definition ~Line 20

**Current Code:**
```python
@dataclass
class MixedICLBatchSpec:
    batch_graphs: int
    num_example: int
    target_index: int
    dtype: torch.dtype = torch.long
    device: Optional[torch.device] = None
```

**New Code:**
```python
@dataclass
class MixedICLBatchSpec:
    batch_graphs: int
    target_index: int
    num_example: Optional[int] = None              # Make optional
    min_context_len: Optional[int] = None          # NEW
    max_context_len: Optional[int] = None          # NEW
    dtype: torch.dtype = torch.long
    device: Optional[torch.device] = None
```

**Then update `__getitem__` to handle dynamic context (like dataset.py lines 80-92)**

---

## 🧪 Testing After Changes

```bash
# 1. Unit tests
python test_mixed_structures.py

# 2. Quick training run (100 steps)
python train_mixed.py \
  --seed 999 \
  --context-len 10 \
  --train-size 100 \
  --train-step 100 \
  --batch-size 12

# 3. Check loss values
# MSE loss should be in range [0.0, 1.0]
# Old BCE loss was typically [0.1, 2.0]

# 4. Full training (if tests pass)
sbatch train_mixed_job.sh
```

---

## ⚠️ Common Issues & Solutions

### Issue 1: "y must be float for MSE loss"
**Solution:** Change `y` dtype from `int64` to `float32` in dataset

### Issue 2: "min_context_len is None"
**Solution:** Either set `num_example` (fixed) OR set both `min/max_context_len`

### Issue 3: Training loss suddenly different
**Expected:** MSE loss is typically 0.1-0.3 (vs BCE 0.5-1.5)
**Normal:** This is correct! Lower is better.

### Issue 4: "max_steps not defined"
**Solution:** Add `max_steps=args.train_step` when creating ICLLightningModule

---

## 📊 Expected Results

### Before (Old Code):
```
Epoch 0: train/loss=0.693, train/acc=0.502
Epoch 10: train/loss=0.450, train/acc=0.758
Epoch 50: train/loss=0.180, train/acc=0.920
```

### After (New Code):
```
Epoch 0: train/loss=0.250, train/tv=0.495
Epoch 10: train/loss=0.120, train/tv=0.220
Epoch 50: train/loss=0.040, train/tv=0.082
```

**Key Difference:** Loss values are lower (MSE vs BCE), but behavior should be similar.

---

## 🎯 Recommendation

**Start with Option A (Minimal):**

1. Change 3 lines of code:
   - `y.astype(np.float32)` in dataset
   - Add `max_steps` parameter in training
   - (Loss function already updated in utils/trainer.py)

2. Test training:
   ```bash
   python train_mixed.py --train-step 1000
   ```

3. If working, then consider Option B (Bayesian labels) later

**Timeline:**
- Option A: 30 min coding + 30 min testing = **1 hour**
- Option B: 2 hours coding + 1 hour testing = **3 hours**

---

## 🤔 Decision Point

**Question for you:** Which option do you want to pursue?

**Option A (Fast):**
- ✅ Get mixed training working quickly
- ✅ Compatible with collaborator's code
- ⚠️ Labels still binary (not full feature parity)
- **Recommend:** Do this first, merge, iterate later

**Option B (Complete):**
- ✅ Full feature parity
- ✅ More sophisticated training
- ⚠️ More complex, higher risk
- **Recommend:** Do this after Option A is tested

---

## 📞 Next Steps

Tell me which option you prefer, and I'll:
1. Make the code changes
2. Run the tests
3. Prepare for git commit/push

Which do you want to do first?
