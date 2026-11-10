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

    NOTE (verified with transformers 4.57): flipping attn.is_causal is NOT sufficient,
    because GPT2Model.forward builds an explicit 4D causal mask via create_causal_mask().
    The forward passes below therefore hand GPT2 a ready-made 4D all-zeros additive mask
    (which masking_utils returns as-is), which is what actually disables causality.
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


def _full_or_causal_mask(disable_causal: bool, B: int, L: int, device, dtype):
    """4D additive mask of zeros (full attention) when non-causal; 2D ones (-> GPT2 builds
    its usual causal mask) otherwise."""
    if disable_causal:
        return torch.zeros((B, 1, L, L), device=device, dtype=dtype)
    return torch.ones((B, L), device=device)


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

        self.disable_causal = bool(disable_causal)
        if disable_causal:
            _try_disable_causal_mask(self.gpt2)

        if hasattr(self.gpt2, 'wpe'):
            # wpe is the positional embedding layer in GPT2Model
            nn.init.zeros_(self.gpt2.wpe.weight)
            self.gpt2.wpe.weight.requires_grad = False
            
        self.read_in = nn.Linear(input_dim, n_embd)
        self.read_out = nn.Linear(n_embd, 1)


    def _attention_mask(self, B, L, device, dtype):
        return _full_or_causal_mask(self.disable_causal, B, L, device, dtype)

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
        attn_mask = self._attention_mask(B, L, x.device, inputs_embeds.dtype)
        
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


class NonCausalGPT2CategoricalHead(nn.Module):
    """
    Same as NonCausalGPT2BinaryHead but predicts K classes.
    read_out: (n_embd -> num_classes). Forward returns logits (B, num_classes).
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 3,
        n_embd: int = 256,
        n_layer: int = 6,
        n_head: int = 8,
        dropout: float = 0.0,
        max_seq_len: int = 2048,
        disable_causal: bool = True,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        cfg = GPT2Config(
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            n_positions=max_seq_len,
            n_ctx=max_seq_len,
            resid_pdrop=dropout,
            embd_pdrop=dropout,
            attn_pdrop=dropout,
        )
        self.gpt2 = GPT2Model(cfg)
        self.max_seq_len = max_seq_len
        self.disable_causal = bool(disable_causal)
        if disable_causal:
            _try_disable_causal_mask(self.gpt2)
        if hasattr(self.gpt2, "wpe"):
            nn.init.zeros_(self.gpt2.wpe.weight)
            self.gpt2.wpe.weight.requires_grad = False
        self.read_in = nn.Linear(input_dim, n_embd)
        self.read_out = nn.Linear(n_embd, num_classes)


    def _attention_mask(self, B, L, device, dtype):
        return _full_or_causal_mask(self.disable_causal, B, L, device, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, D). Returns logits (B, num_classes)."""
        if x.dim() != 3:
            raise ValueError(f"Expected x (B,L,D), got {tuple(x.shape)}")
        x = x.float()
        B, L, _ = x.shape
        inputs_embeds = self.read_in(x)
        attn_mask = self._attention_mask(B, L, x.device, inputs_embeds.dtype)
        position_ids = torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1)
        position_ids = torch.clamp(position_ids, 0, self.max_seq_len - 1)
        out = self.gpt2(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask,
            position_ids=position_ids,
        )
        h_last = out.last_hidden_state[:, -1, :]
        logits = self.read_out(h_last)
        return logits


# ---------------------------------------------------------------------------
# Theory architecture (Theorem 4.1): two-layer, single-head softmax transformer
# ---------------------------------------------------------------------------


class _SelfMaskedAttention(nn.Module):
    """
    Multi-head (default single-head) softmax attention with the "standard masking"
    of Section 4: a token never attends to itself. No output projection, no bias,
    matching Attn(X) = X + V X softmax[(KX)^T (QX)].
    """

    def __init__(self, dim: int, n_head: int = 1) -> None:
        super().__init__()
        if dim % n_head != 0:
            raise ValueError(f"dim={dim} must be divisible by n_head={n_head}")
        self.dim = dim
        self.n_head = n_head
        self.head_dim = dim // n_head
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        B, L, D = h.shape
        H, Dh = self.n_head, self.head_dim
        q = self.q(h).view(B, L, H, Dh).transpose(1, 2)  # (B,H,L,Dh)
        k = self.k(h).view(B, L, H, Dh).transpose(1, 2)
        v = self.v(h).view(B, L, H, Dh).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-1, -2)) / (Dh ** 0.5)  # (B,H,L,L)
        if L > 1:
            eye = torch.eye(L, dtype=torch.bool, device=h.device)
            scores = scores.masked_fill(eye, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)  # (B,H,L,Dh)
        return out.transpose(1, 2).reshape(B, L, D)


class _TheoryBlock(nn.Module):
    """TF(X) = FF[Attn(X)] with FF(X) = X + W2 relu(W1 X). No LayerNorm, no dropout."""

    def __init__(self, dim: int, n_head: int, ffn_mult: int, layernorm: bool) -> None:
        super().__init__()
        self.attn = _SelfMaskedAttention(dim, n_head)
        self.w1 = nn.Linear(dim, ffn_mult * dim)
        self.w2 = nn.Linear(ffn_mult * dim, dim)
        self.ln1 = nn.LayerNorm(dim) if layernorm else nn.Identity()
        self.ln2 = nn.LayerNorm(dim) if layernorm else nn.Identity()

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        h = h + self.attn(self.ln1(h))
        h = h + self.w2(torch.relu(self.w1(self.ln2(h))))
        return h


class TheoryTransformerBinaryHead(nn.Module):
    """
    Faithful implementation of the architecture in Theorem 4.1 for binary nodes:

      f(X) = Linear[Read(TF2(TF1(X)))]

    - input tokens are one-hot encoded: values block (N*d, masked entries -> 0),
      target-node indicator p (N), and a query flag (1) distinguishing the test
      token (the role of p_q vs p in eq. (3.1)).
    - a linear read-in maps that to width n_embd (default (2N+1)*d as in the theorem),
    - n_layer (default 2) blocks of single-head self-masked softmax attention and a
      square ReLU feed-forward layer, both with skip connections,
    - no LayerNorm, no dropout, no positional embedding (the context is a set),
    - Read = last token, Linear -> scalar logit for P(X_t = 1).

    Takes exactly the same (B, L, N+1) integer input as NonCausalGPT2BinaryHead.
    """

    def __init__(
        self,
        input_dim: int,          # N + 1 (same convention as the GPT2 head)
        num_classes: int = 2,
        n_embd: Optional[int] = None,
        n_layer: int = 2,
        n_head: int = 1,
        ffn_mult: int = 1,
        layernorm: bool = False,
        **unused,                # swallow gpt2-only kwargs (dropout, max_seq_len, ...)
    ) -> None:
        super().__init__()
        self.num_nodes = int(input_dim) - 1
        self.num_classes = int(num_classes)
        N, d = self.num_nodes, self.num_classes
        if n_embd is None:
            n_embd = (2 * N + 1) * d
        self.n_embd = int(n_embd)
        self.onehot_dim = N * d + N + 1
        self.read_in = nn.Linear(self.onehot_dim, self.n_embd)
        self.blocks = nn.ModuleList(
            [_TheoryBlock(self.n_embd, n_head, ffn_mult, layernorm) for _ in range(n_layer)]
        )
        self.read_out = nn.Linear(self.n_embd, 1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """(B, L, N+1) ints (-1 = masked) -> (B, L, N*d + N + 1) one-hot floats."""
        B, L, _ = x.shape
        N, d = self.num_nodes, self.num_classes
        vals = x[..., :N].long()                       # (B,L,N) in {-1,0,..,d-1}
        tgt = x[..., N].long()                         # (B,L)
        valid = (vals >= 0)
        oh = torch.nn.functional.one_hot(vals.clamp(min=0), d).float()  # (B,L,N,d)
        oh = oh * valid.unsqueeze(-1).float()
        oh = oh.reshape(B, L, N * d)
        p = torch.nn.functional.one_hot(tgt.clamp(0, N - 1), N).float()  # (B,L,N)
        q = torch.zeros((B, L, 1), dtype=torch.float32, device=x.device)
        q[:, -1, 0] = 1.0                              # query flag on the test token
        return torch.cat([oh, p, q], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"Expected x (B,L,D), got {tuple(x.shape)}")
        h = self.read_in(self.encode(x))
        for blk in self.blocks:
            h = blk(h)
        return self.read_out(h[:, -1, :]).squeeze(-1)


class TheoryTransformerCategoricalHead(TheoryTransformerBinaryHead):
    """
    K-class version of TheoryTransformerBinaryHead (Theorem 4.1 architecture).

    Same one-hot encoding, read-in, 2 self-masked single-head softmax blocks with a
    ReLU feed-forward, no LayerNorm/dropout/positions. Default width (2N+1)K.
    Read-out is Linear(n_embd -> K); forward returns logits (B, K).
    """

    def __init__(self, input_dim: int, num_classes: int = 3, **kwargs) -> None:
        super().__init__(input_dim=input_dim, num_classes=num_classes, **kwargs)
        self.read_out = nn.Linear(self.n_embd, self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"Expected x (B,L,D), got {tuple(x.shape)}")
        h = self.read_in(self.encode(x))
        for blk in self.blocks:
            h = blk(h)
        return self.read_out(h[:, -1, :])


def build_categorical_model(arch: str, input_dim: int, num_classes: int, **kwargs) -> nn.Module:
    """Factory for K-class heads. arch in {"gpt2", "theory"}."""
    if arch == "gpt2":
        gpt2_keys = {"n_embd", "n_layer", "n_head", "dropout", "max_seq_len", "disable_causal"}
        return NonCausalGPT2CategoricalHead(
            input_dim=input_dim, num_classes=num_classes,
            **{k: v for k, v in kwargs.items() if k in gpt2_keys},
        )
    if arch == "theory":
        return TheoryTransformerCategoricalHead(input_dim=input_dim, num_classes=num_classes, **kwargs)
    raise ValueError(f"Unknown arch {arch!r}; expected 'gpt2' or 'theory'")


def build_binary_model(arch: str, input_dim: int, **kwargs) -> nn.Module:
    """Factory used by the Lightning modules. arch in {"gpt2", "theory"}."""
    if arch == "gpt2":
        gpt2_keys = {"n_embd", "n_layer", "n_head", "dropout", "max_seq_len", "disable_causal"}
        return NonCausalGPT2BinaryHead(
            input_dim=input_dim, **{k: v for k, v in kwargs.items() if k in gpt2_keys}
        )
    if arch == "theory":
        return TheoryTransformerBinaryHead(input_dim=input_dim, **kwargs)
    raise ValueError(f"Unknown arch {arch!r}; expected 'gpt2' or 'theory'")
