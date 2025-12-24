# multigraph_sampler.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

from .bn_template import BNTemplate
from .binary_bn import BNError


def _parent_config_from_X(X: np.ndarray, parent_idx: np.ndarray) -> np.ndarray:
    """
    X: (T, B, N) uint8 in {0,1}
    parent_idx: (k,) indices into N
    returns cfg: (T, B) int64 in [0, 2^k)
    """
    k = int(parent_idx.size)
    if k == 0:
        return np.zeros((X.shape[0], X.shape[1]), dtype=np.int64)

    pv = X[:, :, parent_idx]  # (T, B, k)
    weights = (1 << np.arange(k, dtype=np.int64))[None, None, :]  # (1,1,k)
    return (pv.astype(np.int64) * weights).sum(axis=2)  # (T, B)

def sample_target_indices(
    rng: np.random.Generator,
    batch_size: int,
    max_target: int,
) -> np.ndarray:
    """
    Sample per-sample target indices:
        t_b ~ Uniform({0, ..., max_target})
    Returns shape (B,) int64.
    """
    return rng.integers(
        low=0,
        high=max_target + 1,
        size=batch_size,
        dtype=np.int64,
    )


def sample_many_graphs(
    template: BNTemplate,
    p1_list: List[np.ndarray],   # each (G, 2^k_i)
    graph_ids: np.ndarray,       # (T,) int64, selecting graphs
    num_examples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Vectorized ancestral sampling for T selected graphs in parallel.

    Returns:
        X: (T, E, N) uint8, where:
            T = len(graph_ids)
            E = num_examples
            N = template.num_nodes
        Node order is template.topo_nodes.
    """
    graph_ids = np.asarray(graph_ids, dtype=np.int64).reshape(-1)
    T = int(graph_ids.shape[0])
    E = int(num_examples)
    N = template.num_nodes

    X = np.empty((T, E, N), dtype=np.uint8)

    for i, parents in enumerate(template.parent_idx):
        cfg = _parent_config_from_X(X, parents)  # (T, E)
        # Select the CPT rows for the chosen graphs: (T, 2^k)
        p1 = p1_list[i][graph_ids, :]  # advanced indexing
        # p = p1[t, cfg[t,e]]
        p = np.take_along_axis(p1, cfg, axis=1)  # (T, E)
        u = rng.random((T, E))
        X[:, :, i] = (u < p).astype(np.uint8)

    return X


@dataclass
class MultiGraphBatchSampler:
    template: BNTemplate
    p1_list: List[np.ndarray]  # each (G, 2^k_i)
    seed: int = 0

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self.num_graphs = int(self.p1_list[0].shape[0])

        if len(self.p1_list) != self.template.num_nodes:
            raise BNError("p1_list length must match number of nodes in template.")

        for i, parents in enumerate(self.template.parent_idx):
            k = int(parents.size)
            K = 1 << k
            if self.p1_list[i].shape[1] != K:
                raise BNError(f"Node {i} expected CPT width {K}, got {self.p1_list[i].shape[1]}.")

    def batch(
        self,
        batch_graphs: int,
        num_examples: int,
        target_index: int,
        return_unmasked: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], int]:
        """
        target_index is interpreted as MAX target index (T).
        We sample one t ~ Uniform({0, ..., T}) and use it for the whole batch.

        Returns:
            graph_ids: (B,) int64
            X_masked: (B, E, N) uint8   (masked from t onward)
            X_full (optional): (B, E, N) uint8
            t: int (the sampled target index used for masking)
        """
        B = int(batch_graphs)
        E = int(num_examples)
        N = int(self.template.num_nodes)

        T = int(target_index)
        if not (0 <= T < N):
            raise BNError(f"target_index (max) must be in [0, {N-1}], got {T}.")

        # Sample a SINGLE target index for the whole batch
        t = int(self.rng.integers(0, T + 1))

        graph_ids = self.rng.integers(0, self.num_graphs, size=B, dtype=np.int64)
        X_full = sample_many_graphs(
            template=self.template,
            p1_list=self.p1_list,
            graph_ids=graph_ids,
            num_examples=E,
            rng=self.rng,
        )

        X_masked = X_full.copy()
        X_masked[:, :, t:] = 0  # mask target and all future nodes (inclusive)

        if return_unmasked:
            return graph_ids, X_masked, X_full, t

        return graph_ids, X_masked, None, t