"""
Mixed-structure ICL dataset that samples from multiple graph topologies.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Iterator, Optional, List

import numpy as np
import torch
from torch.utils.data import IterableDataset

from .multigraph_sampler import sample_many_graphs
from .bn_template import BNTemplate
from .binary_bn import BNError


@dataclass
class MixedICLBatchSpec:
    """Batch specification for mixed-structure training."""
    batch_graphs: int              # B (total batch size)
    num_example: int               # number of context examples (L-1)
    target_index: int              # t
    dtype: torch.dtype = torch.long
    device: Optional[torch.device] = None


class MixedGraphICLSequenceDataset(IterableDataset):
    """
    Yields infinite batches where each batch contains samples from multiple graph structures.
    
    For a batch of size B with K structures:
    - Each structure gets approximately B/K samples
    - All structures must have the same number of nodes
    
    Each batch element:
      - sample L = num_example + 1 observations (context + test token)
      - apply masking (context: mask t+1..N-1, test: mask t..N-1)
      - append target_index as last feature dimension, so D = N + 1

    Output:
      batch["x"]: (B, L, N+1) masked, last dim is target index
      batch["y"]: (B, L, N)   unmasked ground truth node values
      batch["graph_id"]: (B,) global graph ids across all structures
      batch["structure_id"]: (B,) which structure each sample comes from
      batch["topo_nodes"]: list[str] node names in column order (same for all structures)
      batch["target_index"]: int
    """

    def __init__(
        self,
        templates: List[BNTemplate],           # One template per structure
        p1_lists: List[List[np.ndarray]],      # One p1_list per structure
        structure_names: List[str],            # Names of structures
        seed: int,
        spec: MixedICLBatchSpec,
        return_full: bool = True,
    ) -> None:
        super().__init__()
        
        if len(templates) != len(p1_lists) != len(structure_names):
            raise BNError("templates, p1_lists, and structure_names must have same length")
        
        self.num_structures = len(templates)
        self.templates = templates
        self.p1_lists = p1_lists
        self.structure_names = structure_names
        self.spec = spec
        self.return_full = return_full

        self.rng = np.random.default_rng(seed)

        # Verify all structures have same number of nodes
        num_nodes = templates[0].num_nodes
        for i, template in enumerate(templates):
            if template.num_nodes != num_nodes:
                raise BNError(f"All structures must have same number of nodes. "
                            f"Structure {i} has {template.num_nodes}, expected {num_nodes}")
            if len(p1_lists[i]) != num_nodes:
                raise BNError(f"p1_list length for structure {i} must match num_nodes")

        # Get number of graphs per structure
        self.num_graphs_per_structure = [int(p1_list[0].shape[0]) for p1_list in p1_lists]
        
        # Verify target index
        t = int(spec.target_index)
        if not (0 <= t < num_nodes):
            raise BNError(f"target_index must be in [0, {num_nodes - 1}]")

        # Use topo_nodes from first template (all should be same)
        self.topo_nodes = list(templates[0].topo_nodes)
        self.num_nodes = num_nodes

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        B = int(self.spec.batch_graphs)
        L = int(self.spec.num_example) + 1
        t = int(self.spec.target_index)
        N = self.num_nodes

        dtype = self.spec.dtype
        device = self.spec.device

        # Compute samples per structure (roughly equal split)
        samples_per_structure = []
        remaining = B
        for i in range(self.num_structures):
            if i == self.num_structures - 1:
                # Last structure gets remaining samples
                samples_per_structure.append(remaining)
            else:
                # Equal split, rounded down
                n_samples = B // self.num_structures
                samples_per_structure.append(n_samples)
                remaining -= n_samples

        while True:
            # Storage for batch
            X_full_batch = []
            X_mask_batch = []
            y_batch = []
            graph_id_batch = []
            structure_id_batch = []

            # Sample from each structure
            for struct_idx in range(self.num_structures):
                B_struct = samples_per_structure[struct_idx]
                
                # Sample graph IDs for this structure
                graph_ids = self.rng.integers(
                    0, self.num_graphs_per_structure[struct_idx], 
                    size=B_struct, 
                    dtype=np.int64
                )

                # Sample L observations per graph: (B_struct, L, N)
                X_full = sample_many_graphs(
                    template=self.templates[struct_idx],
                    p1_list=self.p1_lists[struct_idx],
                    graph_ids=graph_ids,
                    num_examples=L,
                    rng=self.rng,
                )

                # Masked copy
                X_mask = X_full.copy()
                
                # Context rows: mask strictly future nodes (t+1:)
                if t + 1 < N:
                    X_mask[:, : L - 1, t + 1 :] = 0

                # Test row: mask target and future (t:)
                X_mask[:, L - 1, t:] = 0

                # Extract targets (last row, target column)
                y = X_full[:, L - 1, t].astype(np.int64)  # (B_struct,)

                # Store
                X_full_batch.append(X_full)
                X_mask_batch.append(X_mask)
                y_batch.append(y)
                graph_id_batch.append(graph_ids)
                structure_id_batch.append(np.full(B_struct, struct_idx, dtype=np.int64))

            # Concatenate all structures
            X_full_combined = np.concatenate(X_full_batch, axis=0)  # (B, L, N)
            X_mask_combined = np.concatenate(X_mask_batch, axis=0)  # (B, L, N)
            y_combined = np.concatenate(y_batch, axis=0)  # (B,)
            graph_id_combined = np.concatenate(graph_id_batch, axis=0)  # (B,)
            structure_id_combined = np.concatenate(structure_id_batch, axis=0)  # (B,)

            # Shuffle the batch (optional but recommended for mixing)
            shuffle_idx = self.rng.permutation(B)
            X_full_combined = X_full_combined[shuffle_idx]
            X_mask_combined = X_mask_combined[shuffle_idx]
            y_combined = y_combined[shuffle_idx]
            graph_id_combined = graph_id_combined[shuffle_idx]
            structure_id_combined = structure_id_combined[shuffle_idx]

            # Append target index feature as last dimension -> (B, L, N+1)
            tgt = np.full((B, L, 1), t, dtype=np.int64)
            X_out = np.concatenate([X_mask_combined.astype(np.int64), tgt], axis=2)

            batch: Dict[str, Any] = {
                "x": torch.as_tensor(X_out, dtype=dtype, device=device),  # (B, L, N+1)
                "y": torch.as_tensor(y_combined, dtype=dtype, device=device),  # (B,)
                "graph_id": torch.as_tensor(graph_id_combined, dtype=torch.long, device=device),  # (B,)
                "structure_id": torch.as_tensor(structure_id_combined, dtype=torch.long, device=device),  # (B,)
                "topo_nodes": self.topo_nodes,
                "target_index": t,
            }

            if self.return_full:
                batch["full"] = torch.as_tensor(
                    X_full_combined.astype(np.int64), 
                    dtype=dtype, 
                    device=device
                )  # (B, L, N)

            yield batch
