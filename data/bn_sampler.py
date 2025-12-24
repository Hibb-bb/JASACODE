# bn_sampler.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional, Tuple

import numpy as np

from .binary_bn import CompiledBinaryBayesNet, BNError


def _compute_parent_config(batch_X: np.ndarray, parent_idx: np.ndarray) -> np.ndarray:
    """
    Compute parent bit-config ids for each row in the batch.

    parent_idx: shape (k,), indices into columns of batch_X
    Returns cfg: shape (B,), dtype=int64, in [0, 2^k)
    """
    if parent_idx.size == 0:
        # No parents: single config 0 for all rows
        return np.zeros(batch_X.shape[0], dtype=np.int64)

    parents_vals = batch_X[:, parent_idx]  # (B, k) in {0,1}
    # cfg = sum_{j} parents_vals[:, j] * 2^j
    weights = (1 << np.arange(parent_idx.size, dtype=np.int64))[None, :]  # (1, k)
    cfg = (parents_vals.astype(np.int64) * weights).sum(axis=1)
    return cfg


@dataclass
class BatchSampler:
    """
    High-throughput ancestral sampler for CompiledBinaryBayesNet.

    - Produces X in topological node order (compiled_bn.topo_nodes).
    - All values are uint8 in {0,1}.
    """
    bn: CompiledBinaryBayesNet
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def sample(self, batch_size: int) -> np.ndarray:
        if batch_size <= 0:
            raise BNError("batch_size must be positive.")
        B = int(batch_size)
        N = self.bn.num_nodes
        X = np.empty((B, N), dtype=np.uint8)

        # Sample nodes in topological order
        for i, spec in enumerate(self.bn.specs):
            cfg = _compute_parent_config(X, spec.parents)  # shape (B,)
            p = spec.p1[cfg]  # P(X_i=1 | cfg), shape (B,)
            u = self.rng.random(B)
            X[:, i] = (u < p).astype(np.uint8)

        return X

    def sample_with_joint_index(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (X, idx) where idx is the joint-state index in [0, 2^N).
        idx = sum_{i=0..N-1} X[:, i] * 2^i  (topological node order).
        Useful if you want histograms for TV/KL on small N.
        """
        X = self.sample(batch_size)
        weights = (1 << np.arange(X.shape[1], dtype=np.int64))[None, :]
        idx = (X.astype(np.int64) * weights).sum(axis=1)
        return X, idx


@dataclass
class InfiniteDataLoader:
    """
    Iterable that yields infinite batches from a sampler.

    Example:
        loader = InfiniteDataLoader(sampler, batch_size=100_000)
        X = next(iter(loader))
    """
    sampler: BatchSampler
    batch_size: int

    def __iter__(self) -> Iterator[np.ndarray]:
        while True:
            yield self.sampler.sample(self.batch_size)


def empirical_tv_kl_from_samples(
    idx_true: np.ndarray,
    idx_model: np.ndarray,
    num_states: int,
    eps: float = 1e-12,
) -> Tuple[float, float]:
    """
    Monte Carlo estimate of TV and KL between two discrete distributions over {0..num_states-1},
    using samples of joint indices from each distribution.

    TV(p,q) = 0.5 * sum |p - q|
    KL(p||q) = sum p * log(p/q)

    Notes:
    - With small N (<=10), num_states=2^N is at most 1024, so histogramming is cheap.
    - eps prevents log(0) / division by 0.
    """
    idx_true = np.asarray(idx_true, dtype=np.int64).reshape(-1)
    idx_model = np.asarray(idx_model, dtype=np.int64).reshape(-1)

    p = np.bincount(idx_true, minlength=num_states).astype(np.float64)
    q = np.bincount(idx_model, minlength=num_states).astype(np.float64)

    p /= max(p.sum(), 1.0)
    q /= max(q.sum(), 1.0)

    tv = 0.5 * np.abs(p - q).sum()
    kl = float((p * (np.log(p + eps) - np.log(q + eps))).sum())
    return float(tv), kl
