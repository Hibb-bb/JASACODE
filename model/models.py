# noncausal_gpt2.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2Model


def _try_disable_causal_mask(gpt2: GPT2Model) -> None:
    """
    Best-effort: disable causal masking inside GPT2 attention blocks.

    Transformers has changed APIs across versions. We attempt the most common knobs:
    - attn.is_causal = False
    - attn.bias / masked_bias handling differs by version; we do not rely on it.

    If this fails silently, GPT2 may remain causal. For strict non-causal behavior,
    confirm with a unit test (recommended).
    """
    # GPT2Model.h is ModuleList of GPT2Blocks
    if not hasattr(gpt2, "h"):
        return

    for block in gpt2.h:
        attn = getattr(block, "attn", None)
        if attn is None:
            continue

        # Common in newer versions
        if hasattr(attn, "is_causal"):
            try:
                attn.is_causal = False
            except Exception:
                pass

        # Some versions store on the attention forward
        if hasattr(attn, "attn_dropout") and hasattr(attn, "_attn"):
            # no-op; leaving here as a marker for future custom patches
            pass


class NonCausalGPT2BinaryHead(nn.Module):
    """
    Treat each row in your (B, L, D) matrix as one "token".
    - Project features -> GPT2 hidden size via read_in
    - Run GPT2Model using inputs_embeds
    - Take hidden state at the test token (last position)
    - Predict scalar logit for y in {0,1}

    This matches your ICL format: last row is test token, label is test token's target bit.
    """

    def __init__(
        self,
        input_dim: int,         # D = N+1
        n_embd: int = 256,
        n_layer: int = 6,
        n_head: int = 8,
        dropout: float = 0.0,
        max_seq_len: int = 2048,
        disable_causal: bool = True,
    ) -> None:
        super().__init__()

        cfg = GPT2Config(
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            n_positions=max_seq_len,
            n_ctx=max_seq_len,
            resid_pdrop=dropout,
            embd_pdrop=dropout,
            attn_pdrop=dropout,
            # GPT2 is decoder-only by design; we still patch attention modules below.
        )
        self.gpt2 = GPT2Model(cfg)
        self.max_seq_len = max_seq_len  # Store for position_id clamping

        if disable_causal:
            _try_disable_causal_mask(self.gpt2)

        self.read_in = nn.Linear(input_dim, n_embd)
        self.read_out = nn.Linear(n_embd, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, D) numeric (int/float). We'll cast to float.
        returns logits: (B,)
        """
        if x.dim() != 3:
            raise ValueError(f"Expected x with shape (B,L,D), got {tuple(x.shape)}")

        x = x.float()
        B, L, _ = x.shape

        inputs_embeds = self.read_in(x)                 # (B, L, n_embd)
        attn_mask = torch.ones((B, L), device=x.device) # allow full attention
        
        # Explicitly create position_ids to avoid out-of-bounds errors
        # GPT2 needs position_ids when using inputs_embeds
        # Clamp to max_seq_len to handle sequences longer than training length
        position_ids = torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1)  # (B, L)
        position_ids = torch.clamp(position_ids, 0, self.max_seq_len - 1)  # Clamp to valid range

        out = self.gpt2(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask,
            position_ids=position_ids,
        )
        h = out.last_hidden_state                        # (B, L, n_embd)

        h_last = h[:, -1, :]                             # (B, n_embd) test token
        logits = self.read_out(h_last).squeeze(-1)        # (B,)
        return logits
