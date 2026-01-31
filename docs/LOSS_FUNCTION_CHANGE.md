# Loss Function Change: MSE → L1 (TV Distance)

## Summary

Changed the training loss from **MSE (L2)** to **L1 (MAE/TV)** to match the evaluation metric.

## Motivation

### Before (MSE Loss)
- **Training**: MSE = (p_hat - y)²
- **Evaluation**: TV = |p_hat - y|
- **Problem**: Training optimizes a different metric than what we evaluate on!

### After (L1 Loss)
- **Training**: L1 = |p_hat - y| (same as TV)
- **Evaluation**: TV = |p_hat - y|
- **Benefit**: Training directly optimizes the evaluation metric ✅

## Changes Made

### 1. Modified `utils/trainer.py`

Added `loss_type` parameter to `ICLLightningModule`:

```python
def __init__(
    self,
    input_dim: int,
    init_lr: float = 3e-4,
    weight_decay: float = 1e-2,
    max_steps: int = 100_000,
    warmup_steps: int = 1000,
    min_lr: float = 0.0,
    loss_type: str = "l1",  # NEW: "l1" (TV/MAE) or "mse" (L2)
    **model_kwargs,
)
```

Updated `training_step()` to support both loss types:

```python
def training_step(self, batch, batch_idx: int):
    # ...
    p_hat = torch.sigmoid(logits)
    
    # Compute loss based on loss_type
    if self.loss_type == "l1":
        loss = F.l1_loss(p_hat, y.float())  # TV distance
    elif self.loss_type == "mse":
        loss = F.mse_loss(p_hat, y.float())  # MSE
    
    # Always log TV for comparison
    tv = torch.abs(p_hat - y.float()).mean()
    # ...
```

### 2. Modified `train_mixed.py`

Set default to L1 loss:

```python
lit = ICLLightningModule(
    input_dim=input_dim,
    init_lr=args.init_lr,
    weight_decay=1e-2,
    max_steps=args.train_step,
    warmup_steps=args.warmup_steps,
    min_lr=args.min_lr,
    loss_type="l1",  # NEW: Use L1 loss to match evaluation
    n_embd=256,
    n_layer=12,
    n_head=8,
    dropout=0.1,
    max_seq_len=max_seq_len,
    disable_causal=True,
)
```

## Usage

### Use L1 loss (default, recommended):
```python
lit = ICLLightningModule(..., loss_type="l1")
```

### Use MSE loss (old behavior):
```python
lit = ICLLightningModule(..., loss_type="mse")
```

## Expected Impact

### Benefits:
1. **Better alignment**: Training metric = Evaluation metric
2. **Potentially better performance**: Directly optimizing TV distance
3. **More robust**: L1 less sensitive to outliers than MSE

### Trade-offs:
1. **Slower convergence**: L1 gradients are constant (not scaled by error)
2. **Different optimal LR**: May need to retune learning rate

## Backward Compatibility

- Old checkpoints trained with MSE will still load
- You can switch back to MSE by setting `loss_type="mse"`
- Both loss types are supported for flexibility

## Testing Plan

1. Train new models with L1 loss (default)
2. Compare TV distance on evaluation:
   - Old models (MSE loss)
   - New models (L1 loss)
3. Expected: L1-trained models should have lower TV distance

## Date

January 11, 2026
