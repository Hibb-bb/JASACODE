# train_icl.py
from __future__ import annotations

import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from models import NonCausalGPT2BinaryHead
from bn_template import compile_template_from_structure, init_graph_params_beta
from torch_icL_bn_dataset import ICLBatchSpec, MultiGraphICLSequenceDataset  # the version that returns x:(B,L,D), y:(B,)


class ICLLightningModule(pl.LightningModule):
    def __init__(
        self,
        input_dim: int,
        init_lr: float = 3e-4,
        weight_decay: float = 1e-2,
        **model_kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.model = NonCausalGPT2BinaryHead(input_dim=input_dim, **model_kwargs)
        self.init_lr = init_lr
        self.weight_decay = weight_decay

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx: int):
        x = batch["x"]  # (B, L, D)
        y = batch["y"]  # (B,)
        logits = self(x)

        loss = F.binary_cross_entropy_with_logits(logits, y.float())

        with torch.no_grad():
            pred = (torch.sigmoid(logits) > 0.5).long()
            acc = (pred == y).float().mean()

        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=False)
        self.log("train/acc", acc, prog_bar=True, on_step=True, on_epoch=False)
        return loss

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.init_lr, weight_decay=self.weight_decay)
        return opt