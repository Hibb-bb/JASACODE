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



def init_graph_params_uniform(
    template: BNTemplate,
    num_graphs: int,
    seed: int | None = None,
) -> List[np.ndarray]:
    """
    Returns per-node CPT tables for many graphs with i.i.d. Uniform(0,1) entries.

    Output:
        p1_list: length = num_nodes
        p1_list[i] has shape (G, 2^k_i),
        where k_i is the in-degree of node i (in topo order).
    """
    rng = np.random.default_rng(seed)
    G = int(num_graphs)

    p1_list: List[np.ndarray] = []
    for parents in template.parent_idx:
        k = int(parents.size)
        K = 1 << k
        p = 0.1 + 0.8 * rng.random(size=(G, K)).astype(np.float64)
        p1_list.append(p)

    return p1_list


def init_graph_params_beta(
    template: BNTemplate,
    num_graphs: int,
    alpha: float = 2.0,
    seed: int | None = None,
) -> List[np.ndarray]:
    """
    Returns per-node CPT tables for many graphs with i.i.d. Beta(alpha, alpha) entries.

    Intuition:
      - alpha > 1 concentrates around 0.5 (\"flatter\" / less extreme CPTs)
      - alpha = 1 reduces to Uniform(0,1)
      - alpha < 1 pushes mass toward 0 and 1 (\"peaky\" / more extreme CPTs)

    Output schema matches init_graph_params_uniform:
        p1_list[i] has shape (G, 2^k_i), where k_i is in-degree of node i.
    """
    if alpha <= 0:
        raise BNError(f"alpha must be > 0, got {alpha}")
    rng = np.random.default_rng(seed)
    G = int(num_graphs)

    p1_list: List[np.ndarray] = []
    for parents in template.parent_idx:
        k = int(parents.size)
        K = 1 << k
        p = rng.beta(alpha, alpha, size=(G, K)).astype(np.float64)
        p1_list.append(p)

    return p1_list
