# categorical_template.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .categorical_bn import CategoricalBayesNet, CompiledCategoricalBayesNet, BNError


@dataclass(frozen=True)
class CategoricalTemplate:
    """Structure-only template for categorical BN (cardinality K, same for all nodes)."""
    topo_nodes: List[str]
    parent_idx: List[np.ndarray]  # each (k,) int64
    num_nodes: int
    cardinality: int  # K


def compile_template_from_categorical(bn: CategoricalBayesNet) -> CategoricalTemplate:
    """Extract structure (topo order + parent indices) from a CategoricalBayesNet."""
    compiled: CompiledCategoricalBayesNet = bn.compile()
    parent_idx = [spec.parents.copy() for spec in compiled.specs]
    return CategoricalTemplate(
        topo_nodes=compiled.topo_nodes,
        parent_idx=parent_idx,
        num_nodes=compiled.num_nodes,
        cardinality=compiled.cardinality,
    )


def init_graph_params_categorical(
    template: CategoricalTemplate,
    num_graphs: int,
    seed: int | None = None,
    dirichlet_alpha: float = 1.0,
) -> List[np.ndarray]:
    """
    Returns per-node CPT tables for many graphs. Each CPT row is Dirichlet(α,...,α) over K classes.

    dirichlet_alpha: Symmetric concentration per class. α=1 → uniform on simplex (exchangeable).
    α < 1 → more peaked / "less uniform" rows; α > 1 → concentrated near (1/K,...,1/K).

    Output:
        cpt_list: length = num_nodes
        cpt_list[i] has shape (G, K^k_i, K) with K = template.cardinality.
    """
    if dirichlet_alpha <= 0:
        raise BNError(f"dirichlet_alpha must be > 0, got {dirichlet_alpha}")
    rng = np.random.default_rng(seed)
    G = int(num_graphs)
    K = template.cardinality
    alpha_vec = np.full(K, float(dirichlet_alpha), dtype=np.float64)

    cpt_list: List[np.ndarray] = []
    for parents in template.parent_idx:
        k = int(parents.size)
        num_configs = K ** k
        # (G, num_configs, K) - each row sums to 1
        p = rng.dirichlet(alpha=alpha_vec, size=(G, num_configs))
        cpt_list.append(p.astype(np.float64))

    return cpt_list
