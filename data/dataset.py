from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Iterator, Optional

import numpy as np
import torch
from torch.utils.data import IterableDataset

from .multigraph_sampler import sample_many_graphs
from .bn_template import BNTemplate
from .binary_bn import BNError


@dataclass
class ICLBatchSpec:
    batch_graphs: int              # B
    num_example: int               # number of context examples (L-1)
    target_index: int              # t
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
      batch["y"]: (B, L, N)   unmasked ground truth node values (optional but useful)
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

        t = int(spec.target_index)
        if not (0 <= t < template.num_nodes):
            raise BNError(f"target_index must be in [0, {template.num_nodes - 1}]")

        self.topo_nodes = list(template.topo_nodes)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        B = int(self.spec.batch_graphs)
        L = int(self.spec.num_example) + 1
        t = int(self.spec.target_index)
        N = int(self.template.num_nodes)

        dtype = self.spec.dtype
        device = self.spec.device

        while True:
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
            y = X_full[:, L - 1, t].astype(np.int64)  # (B,)
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
                "y": torch.as_tensor(y, dtype=dtype, device=device),         # (B,)
            }

            if self.return_full:
                batch["full"] = torch.as_tensor(X_full.astype(np.int64), dtype=dtype, device=device)  # (B, L, N)

            yield batch
