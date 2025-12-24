# bn_template.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Optional
import numpy as np

from .binary_bn import BinaryBayesNet, CompiledBinaryBayesNet, BNError

@dataclass(frozen=True)
class BNTemplate:
    topo_nodes: list[str]
    parent_idx: list[np.ndarray]
    num_nodes: int


def compile_template_from_structure(bn: BinaryBayesNet) -> BNTemplate:
    """
    Compile only the structure (topological order + parent indices).

    Dummy CPTs (0.5) are injected only to satisfy compilation requirements.
    """
    # Inject dummy CPTs if missing
    for node in bn.nodes:
        try:
            bn.get_cpt(node)
        except Exception:
            parents = bn.get_parents(node)
            k = len(parents)
            bn.set_cpt(node, p1=np.full((1 << k,), 0.5, dtype=np.float64))

    compiled: CompiledBinaryBayesNet = bn.compile()

    parent_idx = [spec.parents.copy() for spec in compiled.specs]

    return BNTemplate(
        topo_nodes=compiled.topo_nodes,
        parent_idx=parent_idx,
        num_nodes=compiled.num_nodes,
    )


def init_graph_params_beta(
    template: BNTemplate,
    num_graphs: int,
    mode: str = "easy",
    seed: int = 0,
) -> List[np.ndarray]:
    """
    Returns per-node CPT tables for many graphs.

    Output:
        p1_list: length = num_nodes
        p1_list[i] has shape (G, 2^k_i), where k_i = in-degree of node i (in topo order)
    """
    rng = np.random.default_rng(seed)
    G = int(num_graphs)

    if mode == "easy":
        alpha = 5.0      # concentrates near 0.5 (high entropy, weaker dependencies)
    elif mode == "medium":
        alpha = 1.0      # uniform on [0,1]
    elif mode == "hard":
        alpha = 0.3      # near-deterministic (peaks near 0 and 1)
    else:
        raise ValueError(f"Unknown mode '{mode}'")

    p1_list: List[np.ndarray] = []
    for parents in template.parent_idx:
        k = int(parents.size)
        K = 1 << k
        p = rng.beta(alpha, alpha, size=(G, K)).astype(np.float64)
        p1_list.append(p)

    return p1_list
