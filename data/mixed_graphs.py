"""
Mixed graph structure utilities for training on multiple graph topologies.

All graphs use 5 nodes to maintain consistent input dimensions.
"""
import numpy as np
from .binary_bn import BinaryBayesNet


def random_binary_cpt(
    num_parents: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Returns p1[cfg] = P(X=1 | cfg), length 2^num_parents,
    with entries i.i.d. Uniform(0, 1).
    """
    if rng is None:
        rng = np.random.default_rng()

    K = 1 << num_parents
    p = rng.random(size=K)   # Uniform(0, 1)

    return p.astype(np.float64)


def get_tree_5node(seed=2000):
    r"""
    5-node binary tree structure:
    
         A (root)
        / \
       B   C (depth-1, both children of A)
      / \
     D   E (depth-2, both children of B)
    
    Structure:
    - 1 parent: A
    - 2 depth-1 children: B, C (both children of A)
    - 2 depth-2 children: D, E (both children of same depth-1 node B)
    """
    rng = np.random.default_rng(seed)
    bn = BinaryBayesNet()
    
    # Add nodes
    for n in ["A", "B", "C", "D", "E"]:
        bn.add_node(n)
    
    # Tree edges
    bn.add_edge("A", "B")
    bn.add_edge("A", "C")
    bn.add_edge("B", "D")
    bn.add_edge("B", "E")
    
    # Set parents
    bn.set_parents("A", [])
    bn.set_parents("B", ["A"])
    bn.set_parents("C", ["A"])
    bn.set_parents("D", ["B"])
    bn.set_parents("E", ["B"])
    
    # Random CPTs
    bn.set_cpt("A", random_binary_cpt(0, rng))
    bn.set_cpt("B", random_binary_cpt(1, rng))
    bn.set_cpt("C", random_binary_cpt(1, rng))
    bn.set_cpt("D", random_binary_cpt(1, rng))
    bn.set_cpt("E", random_binary_cpt(1, rng))
    
    return bn


def get_chain_5node(seed=2000):
    r"""
    5-node chain structure:
    
    A -> B -> C -> D -> E
    
    Linear dependency chain.
    """
    rng = np.random.default_rng(seed)
    bn = BinaryBayesNet()
    
    # Add nodes
    for n in ["A", "B", "C", "D", "E"]:
        bn.add_node(n)
    
    # Chain edges
    bn.add_edge("A", "B")
    bn.add_edge("B", "C")
    bn.add_edge("C", "D")
    bn.add_edge("D", "E")
    
    # Set parents
    bn.set_parents("A", [])
    bn.set_parents("B", ["A"])
    bn.set_parents("C", ["B"])
    bn.set_parents("D", ["C"])
    bn.set_parents("E", ["D"])
    
    # Random CPTs
    bn.set_cpt("A", random_binary_cpt(0, rng))
    bn.set_cpt("B", random_binary_cpt(1, rng))
    bn.set_cpt("C", random_binary_cpt(1, rng))
    bn.set_cpt("D", random_binary_cpt(1, rng))
    bn.set_cpt("E", random_binary_cpt(1, rng))
    
    return bn


def get_general_5node(seed=2000):
    r"""
    5-node general DAG structure:
    
    A   B (two independent roots)
     \ / \
      C   |
     / \ /
    D   E
    
    Structure:
    - A, B: independent roots
    - C: depends on A, B
    - D: depends on A, C
    - E: depends on B, C
    """
    rng = np.random.default_rng(seed)
    bn = BinaryBayesNet()
    
    # Add nodes
    for n in ["A", "B", "C", "D", "E"]:
        bn.add_node(n)
    
    # General DAG edges
    bn.add_edge("A", "C")
    bn.add_edge("B", "C")
    bn.add_edge("A", "D")
    bn.add_edge("C", "D")
    bn.add_edge("B", "E")
    bn.add_edge("C", "E")
    
    # Set parents
    bn.set_parents("A", [])
    bn.set_parents("B", [])
    bn.set_parents("C", ["A", "B"])
    bn.set_parents("D", ["A", "C"])
    bn.set_parents("E", ["B", "C"])
    
    # Random CPTs
    bn.set_cpt("A", random_binary_cpt(0, rng))
    bn.set_cpt("B", random_binary_cpt(0, rng))
    bn.set_cpt("C", random_binary_cpt(2, rng))
    bn.set_cpt("D", random_binary_cpt(2, rng))
    bn.set_cpt("E", random_binary_cpt(2, rng))
    
    return bn


def get_mixed_graph_structures(seed=2000):
    """
    Returns a list of 3 different 5-node graph structures.
    
    Returns:
        list[BinaryBayesNet]: [tree, chain, general]
    """
    return [
        get_tree_5node(seed),
        get_chain_5node(seed + 1),
        get_general_5node(seed + 2),
    ]


def get_structure_names():
    """Returns the names of the three structures in order."""
    return ["tree", "chain", "general"]
