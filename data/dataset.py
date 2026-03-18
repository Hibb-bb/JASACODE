from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Iterator, Optional

import numpy as np
import torch
from torch.utils.data import IterableDataset

from .multigraph_sampler import sample_many_graphs, sample_many_graphs_categorical
from .bn_template import BNTemplate
from .categorical_template import CategoricalTemplate
from .binary_bn import BNError


@dataclass
class ICLBatchSpec:
    batch_graphs: int              # B
    target_index: Optional[int] = None  # t. If None, randomly sample per batch
    num_example: Optional[int] = None  # number of context examples (L-1). If None, use min/max.
    min_context_len: Optional[int] = None  # minimum context length (for dynamic sampling)
    max_context_len: Optional[int] = None  # maximum context length (for dynamic sampling)
    dtype: torch.dtype = torch.long
    device: Optional[torch.device] = None


class MultiGraphICLSequenceDataset(IterableDataset):
    """
    Yields infinite batches. Each batch element corresponds to one randomly-chosen graph/task.

    For each graph in the batch:
      - sample L = num_example + 1 observations (context + test token)
      - apply masking:
          context rows: mask columns t+1..N-1
          test row:     mask columns t..N-1
      - append target_index as last feature dimension, so D = N + 1

    Output:
      batch["x"]: (B, L, N+1) masked, last dim is target index
      batch["y"]: (B,) population CPT estimates from context examples (Bayesian estimation with known DAG structure)
      batch["graph_id"]: (B,) graph ids
      batch["topo_nodes"]: list[str] node names in column order
      batch["target_index"]: int
    """

    def __init__(
        self,
        template: BNTemplate,
        p1_list: list[np.ndarray],      # per-node CPT tables, each (G, 2^k_i)
        seed: int,
        spec: ICLBatchSpec,
        return_full: bool = True,
    ) -> None:
        super().__init__()
        self.template = template
        self.p1_list = p1_list
        self.spec = spec
        self.return_full = return_full

        self.rng = np.random.default_rng(seed)

        self.num_graphs = int(p1_list[0].shape[0])
        if len(p1_list) != template.num_nodes:
            raise BNError("p1_list length must match template.num_nodes")

        # Validate target_index if provided, otherwise will sample randomly per batch
        if spec.target_index is not None:
            t = int(spec.target_index)
            if not (0 <= t < template.num_nodes):
                raise BNError(f"target_index must be in [0, {template.num_nodes - 1}]")

        self.topo_nodes = list(template.topo_nodes)
        self.num_nodes = template.num_nodes

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        B = int(self.spec.batch_graphs)
        N = int(self.num_nodes)

        dtype = self.spec.dtype
        device = self.spec.device

        while True:
            # Determine target index for this batch (same for all batch elements)
            # Resample t until we get a node with at least one parent (k > 0)
            while True:
                t = int(self.rng.integers(0, N, dtype=np.int64))
                parents_idx = self.template.parent_idx[t]
                k = int(parents_idx.size)
                if k > 0 or k == 0:
                    break
                if all(self.template.parent_idx[i].size == 0 for i in range(N)):
                    raise BNError("All nodes have k=0; cannot resample to k>0")

            # Determine context length for this batch (same for all batch elements)
            if self.spec.num_example is not None:
                # Fixed context length
                num_examples = int(self.spec.num_example)
            elif self.spec.min_context_len is not None and self.spec.max_context_len is not None:
                # Dynamic: randomly sample context length per batch
                num_examples = int(self.rng.integers(
                    self.spec.min_context_len, 
                    self.spec.max_context_len + 1  # inclusive upper bound
                ))
            else:
                raise BNError("Must specify either num_example or (min_context_len, max_context_len)")
            
            L = num_examples + 1  # total sequence length (context + test token)
            
            # Sample B graphs/tasks
            graph_ids = self.rng.integers(0, self.num_graphs, size=B, dtype=np.int64)

            # Sample L observations per graph in parallel: (B, L, N)
            X_full = sample_many_graphs(
                template=self.template,
                p1_list=self.p1_list,
                graph_ids=graph_ids,
                num_examples=L,
                rng=self.rng,
            )

            # Masked copy
            X_mask = X_full.copy()
            
            # Compute population CPT estimate from context examples (Bayesian estimation with known DAG structure)
            # This estimates P(X_t=1 | parent_config) from the context examples, not from ground truth CPT
            m = L - 1  # number of context examples
            K = 1 << k  # number of possible parent configurations

            # Context examples for estimating CPT
            X_ctx = X_full[:, :m, :]  # (B, m, N)
            y_ctx = X_ctx[:, :, t].astype(np.int64)  # (B, m) - target values in context

            # Compute parent configurations for context examples and test token (k > 0 guaranteed by resample loop)
            X_ctx_flat = X_ctx.reshape(B * m, N)
            parents_vals_ctx = X_ctx_flat[:, parents_idx].astype(np.int64)  # (B*m, k)
            weights = (1 << np.arange(k, dtype=np.int64))[None, :]  # (1, k)
            cfg_ctx = (parents_vals_ctx * weights).sum(axis=1).reshape(B, m)  # (B, m)

            # Test token parent configuration
            test_prefix = X_full[:, L - 1, :]  # (B, N) - test token values
            parents_vals_test = test_prefix[:, parents_idx].astype(np.int64)  # (B, k)
            cfg_test = (parents_vals_test * weights).sum(axis=1)  # (B,)
            
            # Estimate CPT from context examples using Beta prior (alpha=beta=0.0 = maximum likelihood)
            # For each batch element, estimate P(X_t=1 | cfg) from context counts
            y = np.empty((B,), dtype=np.float32)
            alpha, beta = 0.0, 0.0 # Beta prior parameters (0.0 = no smoothing, use 0.5 for Laplace smoothing)
            for i in range(B):
                # Count occurrences of each parent configuration in context
                tot = np.bincount(cfg_ctx[i], minlength=K).astype(np.float64)  # (K,)
                one = np.bincount(cfg_ctx[i], weights=y_ctx[i], minlength=K).astype(np.float64)  # (K,)
                cfg = int(cfg_test[i])
                
                if tot[cfg] > 0:
                    # Bayesian estimate: P(X_t=1 | cfg) = (alpha + count_1) / (alpha + beta + count_total)
                    y[i] = float((alpha + one[cfg]) / (alpha + beta + tot[cfg]))
                else:
                    # No examples with this parent configuration: fall back to marginal P(X_t=1) from context
                    marginal_p = float(y_ctx[i].mean()) if m > 0 else 0.5
                    y[i] = marginal_p
            
            # Context rows: mask strictly future nodes (t+1:)
            if t + 1 < N:
                X_mask[:, : L - 1, t + 1 :] = 0

            # Test row: mask target and future (t:)
            X_mask[:, L - 1, t:] = 0

            # Append target index feature as last dimension -> (B, L, N+1)
            tgt = np.full((B, L, 1), t, dtype=np.int64)
            X_out = np.concatenate([X_mask.astype(np.int64), tgt], axis=2)

            batch: Dict[str, Any] = {
                "x": torch.as_tensor(X_out, dtype=dtype, device=device),          # (B, L, N+1)
                "graph_id": torch.as_tensor(graph_ids, dtype=torch.long, device=device),
                "topo_nodes": self.topo_nodes,
                "target_index": t,
                "y": torch.as_tensor(y, dtype=torch.float32, device=device),         # (B,) - now probabilities
            }

            if self.return_full:
                batch["full"] = torch.as_tensor(X_full.astype(np.int64), dtype=dtype, device=device)  # (B, L, N)

            yield batch


def _parent_config_categorical(
    X_flat: np.ndarray, parent_idx: np.ndarray, K: int
) -> np.ndarray:
    """Parent config id = sum parent_val[j] * K^j. X_flat (R, N), returns (R,) int64."""
    k = int(parent_idx.size)
    if k == 0:
        return np.zeros((X_flat.shape[0],), dtype=np.int64)
    pv = X_flat[:, parent_idx].astype(np.int64)
    powers = np.power(K, np.arange(k, dtype=np.int64))
    return (pv * powers[None, :]).sum(axis=1)


class MultiGraphICLSequenceDatasetCategorical(IterableDataset):
    """
    ICL dataset for 3-class (or K-class) categorical BN.
    Yields x (B, L, N+1) in {0,1,2}, y (B, K) = P(X_t in {0..K-1} | context).
    """

    def __init__(
        self,
        template: CategoricalTemplate,
        cpt_list: list[np.ndarray],  # per-node (G, K^k_i, K)
        seed: int,
        spec: ICLBatchSpec,
        return_full: bool = True,
    ) -> None:
        super().__init__()
        self.template = template
        self.cpt_list = cpt_list
        self.spec = spec
        self.return_full = return_full
        self.rng = np.random.default_rng(seed)
        self.num_graphs = int(cpt_list[0].shape[0])
        self.K = template.cardinality
        if len(cpt_list) != template.num_nodes:
            raise BNError("cpt_list length must match template.num_nodes")
        if spec.target_index is not None:
            t = int(spec.target_index)
            if not (0 <= t < template.num_nodes):
                raise BNError(f"target_index must be in [0, {template.num_nodes - 1}]")
        self.topo_nodes = list(template.topo_nodes)
        self.num_nodes = template.num_nodes

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        B = int(self.spec.batch_graphs)
        N = int(self.num_nodes)
        K = self.K
        dtype = self.spec.dtype
        device = self.spec.device

        while True:
            t = int(self.rng.integers(0, N, dtype=np.int64))
            parents_idx = self.template.parent_idx[t]
            k = int(parents_idx.size)
            num_configs = K ** k

            if self.spec.num_example is not None:
                num_examples = int(self.spec.num_example)
            elif self.spec.min_context_len is not None and self.spec.max_context_len is not None:
                num_examples = int(self.rng.integers(
                    self.spec.min_context_len,
                    self.spec.max_context_len + 1,
                ))
            else:
                raise BNError("Must specify either num_example or (min_context_len, max_context_len)")

            L = num_examples + 1
            graph_ids = self.rng.integers(0, self.num_graphs, size=B, dtype=np.int64)
            X_full = sample_many_graphs_categorical(
                template=self.template,
                cpt_list=self.cpt_list,
                graph_ids=graph_ids,
                num_examples=L,
                rng=self.rng,
            )

            m = L - 1
            X_ctx = X_full[:, :m, :]
            y_ctx = X_ctx[:, :, t].astype(np.int64)

            X_ctx_flat = X_ctx.reshape(B * m, N)
            cfg_ctx = _parent_config_categorical(X_ctx_flat, parents_idx, K).reshape(B, m)
            test_prefix = X_full[:, L - 1, :]
            cfg_test = _parent_config_categorical(test_prefix, parents_idx, K)

            # Bayesian estimate: Dirichlet(alpha,...,alpha) prior, counts from context
            alpha = 0.5
            y_probs = np.empty((B, K), dtype=np.float32)
            for i in range(B):
                tot = np.bincount(cfg_ctx[i], minlength=num_configs).astype(np.float64)
                counts = np.zeros((num_configs, K), dtype=np.float64)
                for v in range(K):
                    counts[:, v] = np.bincount(cfg_ctx[i], weights=(y_ctx[i] == v).astype(np.float64), minlength=num_configs)
                cfg = int(cfg_test[i])
                if tot[cfg] > 0:
                    row = (alpha + counts[cfg, :]) / (alpha * K + tot[cfg])
                else:
                    # No context with this parent config: use marginal over context
                    marginal = (alpha + np.bincount(y_ctx[i], minlength=K).astype(np.float64)) / (alpha * K + m)
                    row = marginal
                y_probs[i] = row.astype(np.float32)

            X_mask = X_full.copy()
            if t + 1 < N:
                X_mask[:, : L - 1, t + 1 :] = 0
            X_mask[:, L - 1, t:] = 0
            tgt = np.full((B, L, 1), t, dtype=np.int64)
            X_out = np.concatenate([X_mask.astype(np.int64), tgt], axis=2)

            batch: Dict[str, Any] = {
                "x": torch.as_tensor(X_out, dtype=dtype, device=device),
                "graph_id": torch.as_tensor(graph_ids, dtype=torch.long, device=device),
                "topo_nodes": self.topo_nodes,
                "target_index": t,
                "y": torch.as_tensor(y_probs, dtype=torch.float32, device=device),
            }
            if self.return_full:
                batch["full"] = torch.as_tensor(X_full, dtype=dtype, device=device)
            yield batch
