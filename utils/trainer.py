# train_icl.py
from __future__ import annotations

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

from model import NonCausalGPT2BinaryHead


class ICLLightningModule(pl.LightningModule):
    def __init__(
        self,
        input_dim: int,
        init_lr: float = 3e-4,
        weight_decay: float = 0.0,
        max_steps: int = 100_000,
        warmup_steps: int = 1000,
        min_lr: float = 0.0,
        **model_kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.model = NonCausalGPT2BinaryHead(input_dim=input_dim, **model_kwargs)
        self.init_lr = init_lr
        self.weight_decay = weight_decay
        self.max_steps = max_steps
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx: int):
        x = batch["x"]  # (B, L, D)
        y = batch["y"]  # (B,) - now probabilities in [0,1]
        logits = self(x)

        # Convert logits to probabilities and compute L1 loss (matching evaluation metric)
        p_hat = torch.sigmoid(logits)  # (B,)
        loss = F.l1_loss(p_hat, y.float())

        with torch.no_grad():
            # Compute TV distance as metric (matching evaluation metric)
            tv = torch.abs(p_hat - y.float()).mean()
        
        # Log learning rate from optimizer (safe access pattern)
        try:
            current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
            self.log("train/lr", current_lr, prog_bar=False, on_step=True, on_epoch=False)
        except (AttributeError, IndexError):
            # Fallback if trainer/optimizer not available yet
            pass
        
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/tv", tv, prog_bar=True, on_step=True, on_epoch=True)  # TV distance metric
        return loss

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.init_lr, weight_decay=self.weight_decay)
        
        # Linear warmup scheduler: linearly increase from near-zero to init_lr over warmup_steps
        warmup_scheduler = LinearLR(
            opt,
            start_factor=1e-8,  # start at very small value (effectively 0, but must be > 0)
            end_factor=1.0,     # end at 1.0 * init_lr = init_lr
            total_iters=self.warmup_steps,
        )
        
        # Cosine annealing scheduler: decay from init_lr to min_lr over remaining steps
        cosine_steps = max(1, self.max_steps - self.warmup_steps)
        cosine_scheduler = CosineAnnealingLR(
            opt,
            T_max=cosine_steps,
            eta_min=self.min_lr,
        )
        
        # Combine: warmup first, then cosine decay
        scheduler = SequentialLR(
            opt,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[self.warmup_steps],
        )
        
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  # update learning rate every step
            },
        }