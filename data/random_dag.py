"""
Random DAG sampling via Erdos-Renyi on a fixed topological order.

Given N nodes (labeled X0, X1, ..., X{N-1}), for each pair (i, j) with i < j,
include edge i -> j independently with probability `edge_prob`.  This always
produces a valid DAG because edges only go from lower to higher index.

A connectivity guarantee ensures every node participates in at least one edge
(either as parent or child), so there are no isolated nodes.
"""
from __future__ import annotations

from typing import Set, Tuple

import numpy as np

from .binary_bn import BinaryBayesNet


def _ensure_no_isolated_nodes(
    num_nodes: int,
    edges: Set[Tuple[int, int]],
    rng: np.random.Generator,
) -> Set[Tuple[int, int]]:
    """
    For every node that has no edge (neither parent nor child),
    add one random edge connecting it.

    Only edges i -> j with i < j are allowed (DAG constraint).
    """
    # Build sets of nodes that participate in at least one edge
    connected = set()
    for i, j in edges:
        connected.add(i)
        connected.add(j)

    for node in range(num_nodes):
        if node in connected:
            continue

        # Node is isolated — pick a random partner and add an edge
        others = [k for k in range(num_nodes) if k != node]
        partner = int(rng.choice(others))

        # Maintain i < j ordering for DAG guarantee
        lo, hi = min(node, partner), max(node, partner)
        edges.add((lo, hi))
        connected.add(lo)
        connected.add(hi)

    return edges


def sample_random_dag(
    num_nodes: int,
    edge_prob: float,
    rng: np.random.Generator,
) -> BinaryBayesNet:
    """
    Sample a random DAG with `num_nodes` nodes using Erdos-Renyi,
    with a guarantee that every node has at least one edge.

    Parameters
    ----------
    num_nodes : int
        Number of nodes (N).
    edge_prob : float
        Probability of including each possible directed edge i -> j (i < j).
    rng : np.random.Generator
        NumPy random generator for reproducibility.

    Returns
    -------
    BinaryBayesNet
        A BayesNet with the random DAG structure and dummy CPTs (0.5).
        Node names are "X0", "X1", ..., "X{N-1}".
        Topological order = index order (guaranteed by construction).
        Every node participates in at least one edge (no isolated nodes).
    """
    if not 0.0 <= edge_prob <= 1.0:
        raise ValueError(f"edge_prob must be in [0, 1], got {edge_prob}")
    if num_nodes < 2:
        raise ValueError(f"num_nodes must be >= 2, got {num_nodes}")

    # 1. Erdos-Renyi: sample edges
    edges: Set[Tuple[int, int]] = set()
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if rng.random() < edge_prob:
                edges.add((i, j))

    # 2. Fix isolated nodes
    edges = _ensure_no_isolated_nodes(num_nodes, edges, rng)

    # 3. Build BinaryBayesNet
    bn = BinaryBayesNet()
    node_names = [f"X{i}" for i in range(num_nodes)]

    for name in node_names:
        bn.add_node(name)

    for i, j in sorted(edges):
        bn.add_edge(node_names[i], node_names[j])

    # Set parents and dummy CPTs
    for name in node_names:
        parents = bn._parents[name]
        bn.set_parents(name, parents)
        k = len(parents)
        bn.set_cpt(name, np.full(1 << k, 0.5, dtype=np.float64))

    return bn
