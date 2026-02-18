"""
ICL dataset that samples a fresh random DAG per batch.

Each batch:
  1. Sample a random Erdos-Renyi DAG (same N nodes, random edges)
  2. Compile to BNTemplate
  3. Sample B random CPT sets
  4. Ancestral-sample L observations per graph
  5. Apply masking and compute Bayesian Y target
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Iterator, Optional

import numpy as np
import torch
from torch.utils.data import IterableDataset

from .random_dag import sample_random_dag
from .bn_template import BNTemplate, compile_template_from_structure, init_graph_params_uniform
from .multigraph_sampler import sample_many_graphs
from .binary_bn import BNError


@dataclass
class RandomDAGBatchSpec:
    """Batch specification for random-DAG training."""
    batch_graphs: int                           # B (batch size)
    num_nodes: int                              # N
    edge_prob: float = 0.5                      # fixed edge probability
    edge_prob_min: Optional[float] = None       # if set, sample p ~ U(min, max) per batch
    edge_prob_max: Optional[float] = None
    num_example: Optional[int] = None           # fixed context length (L-1)
    min_context_len: Optional[int] = None       # dynamic context length range
    max_context_len: Optional[int] = None
    dtype: torch.dtype = torch.long
    device: Optional[torch.device] = None


class RandomDAGICLDataset(IterableDataset):
    """
    Yields infinite batches from freshly sampled random DAGs.

    Every batch gets a brand-new Erdos-Renyi DAG.  All B examples in the
    batch share the same DAG structure but have independent CPTs.

    Output per batch:
        x      : (B, L, N+1)  masked observations + target index
        y      : (B,)         Bayesian CPT estimate (scalar probability)
        graph_id       : (B,)
        topo_nodes     : list[str]
        target_index   : int
        edge_prob_used : float   (the p used for this batch)
    """

    def __init__(
        self,
        seed: int,
        spec: RandomDAGBatchSpec,
        return_full: bool = True,
    ) -> None:
        super().__init__()
        self.spec = spec
        self.return_full = return_full
        self.rng = np.random.default_rng(seed)

        N = spec.num_nodes
        if N < 2:
            raise BNError(f"num_nodes must be >= 2, got {N}")
        self.num_nodes = N

    # ------------------------------------------------------------------ #
    #  helpers                                                             #
    # ------------------------------------------------------------------ #

    def _sample_edge_prob(self) -> float:
        """Return edge probability for this batch."""
        if self.spec.edge_prob_min is not None and self.spec.edge_prob_max is not None:
            return float(self.rng.uniform(self.spec.edge_prob_min, self.spec.edge_prob_max))
        return float(self.spec.edge_prob)

    def _sample_context_len(self) -> int:
        """Return context length (number of examples, L-1) for this batch."""
        if self.spec.num_example is not None:
            return int(self.spec.num_example)
        if self.spec.min_context_len is not None and self.spec.max_context_len is not None:
            return int(self.rng.integers(
                self.spec.min_context_len,
                self.spec.max_context_len + 1,
            ))
        raise BNError("Must specify either num_example or (min_context_len, max_context_len)")

    # ------------------------------------------------------------------ #
    #  main loop                                                           #
    # ------------------------------------------------------------------ #

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        B = int(self.spec.batch_graphs)
        N = self.num_nodes
        dtype = self.spec.dtype
        device = self.spec.device

        while True:
            # 1. Sample edge probability and random DAG
            p_edge = self._sample_edge_prob()
            dag = sample_random_dag(N, p_edge, self.rng)

            # 2. Compile to template
            template: BNTemplate = compile_template_from_structure(dag)

            # 3. Sample B random CPT sets
            p1_list = init_graph_params_uniform(
                template,
                num_graphs=B,
                seed=int(self.rng.integers(0, 2**31)),
            )
            graph_ids = np.arange(B, dtype=np.int64)

            # 4. Sample target and context length
            t = int(self.rng.integers(0, N))
            num_examples = self._sample_context_len()
            L = num_examples + 1  # context + test token

            # 5. Ancestral-sample L observations per graph: (B, L, N)
            X_full = sample_many_graphs(
                template=template,
                p1_list=p1_list,
                graph_ids=graph_ids,
                num_examples=L,
                rng=self.rng,
            )

            # 6. Masking
            X_mask = X_full.copy()
            if t + 1 < N:
                X_mask[:, :L - 1, t + 1:] = 0   # context: mask future
            X_mask[:, L - 1, t:] = 0             # test: mask target + future

            # 7. Compute Bayesian CPT estimate (Y) from context
            m = L - 1
            parents_idx = template.parent_idx[t]
            k = int(parents_idx.size)
            K = 1 << k

            X_ctx = X_full[:, :m, :]                          # (B, m, N)
            y_ctx = X_ctx[:, :, t].astype(np.int64)           # (B, m)

            if k == 0:
                cfg_ctx = np.zeros((B, m), dtype=np.int64)
                cfg_test = np.zeros((B,), dtype=np.int64)
            else:
                X_ctx_flat = X_ctx.reshape(B * m, N)
                parents_vals = X_ctx_flat[:, parents_idx].astype(np.int64)
                weights = (1 << np.arange(k, dtype=np.int64))[None, :]
                cfg_ctx = (parents_vals * weights).sum(axis=1).reshape(B, m)

                test_vals = X_full[:, L - 1, parents_idx].astype(np.int64)
                cfg_test = (test_vals * weights).sum(axis=1)

            # MLE estimate per example
            y = np.empty((B,), dtype=np.float32)
            alpha, beta = 0.0, 0.0
            for i in range(B):
                tot = np.bincount(cfg_ctx[i], minlength=K).astype(np.float64)
                one = np.bincount(cfg_ctx[i], weights=y_ctx[i], minlength=K).astype(np.float64)
                cfg = int(cfg_test[i])
                if tot[cfg] > 0:
                    y[i] = float((alpha + one[cfg]) / (alpha + beta + tot[cfg]))
                else:
                    marginal_p = float(y_ctx[i].mean()) if m > 0 else 0.5
                    y[i] = marginal_p

            # 8. Append target index as last feature → (B, L, N+1)
            tgt = np.full((B, L, 1), t, dtype=np.int64)
            X_out = np.concatenate([X_mask.astype(np.int64), tgt], axis=2)

            batch: Dict[str, Any] = {
                "x": torch.as_tensor(X_out, dtype=dtype, device=device),
                "y": torch.as_tensor(y, dtype=torch.float32, device=device),
                "graph_id": torch.as_tensor(graph_ids, dtype=torch.long, device=device),
                "topo_nodes": list(template.topo_nodes),
                "target_index": t,
                "edge_prob_used": p_edge,
            }

            if self.return_full:
                batch["full"] = torch.as_tensor(
                    X_full.astype(np.int64), dtype=dtype, device=device,
                )

            yield batch
