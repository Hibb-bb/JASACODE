"""
Mixed graph structure utilities for training on multiple graph topologies.

Provides both 5-node and 7-node versions of each structure.
get_mixed_graph_structures() returns the 7-node versions used for training.
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


# ============================================================
#  5-node structures
# ============================================================

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
    5-node general DAG structure (matches graphs.py get_general):
    
    A   B (two independent roots)
     \ /
      C
     / \
    D   |
    |   |
    E <-+
    
    Structure:
    - A, B: independent roots
    - C: depends on A, B
    - D: depends on A, C
    - E: depends on D
    """
    rng = np.random.default_rng(seed)
    bn = BinaryBayesNet()
    
    # Add nodes
    for n in ["A", "B", "C", "D", "E"]:
        bn.add_node(n)
    
    # General DAG edges (matches graphs.py get_general)
    bn.add_edge("A", "C")
    bn.add_edge("B", "C")
    bn.add_edge("C", "D")
    bn.add_edge("A", "D")
    bn.add_edge("D", "E")
    
    # Set parents
    bn.set_parents("A", [])
    bn.set_parents("B", [])
    bn.set_parents("C", ["A", "B"])
    bn.set_parents("D", ["A", "C"])
    bn.set_parents("E", ["D"])
    
    # Random CPTs
    bn.set_cpt("A", random_binary_cpt(0, rng))
    bn.set_cpt("B", random_binary_cpt(0, rng))
    bn.set_cpt("C", random_binary_cpt(2, rng))
    bn.set_cpt("D", random_binary_cpt(2, rng))
    bn.set_cpt("E", random_binary_cpt(1, rng))
    
    return bn


# ============================================================
#  7-node structures (matching graphs.py)
# ============================================================

def get_tree_7node(seed=2000):
    r"""
    7-node binary tree structure (matches graphs.py get_tree):
    
           A (root)
          / \
         B   C
        / \  / \
       D  E F   G
    
    Structure:
    - A: root (0 parents)
    - B, C: depth-1 children of A (1 parent each)
    - D, E: depth-2 children of B (1 parent each)
    - F, G: depth-2 children of C (1 parent each)
    """
    rng = np.random.default_rng(seed)
    bn = BinaryBayesNet()
    
    # Add nodes
    for n in ["A", "B", "C", "D", "E", "F", "G"]:
        bn.add_node(n)
    
    # Tree edges
    bn.add_edge("A", "B")
    bn.add_edge("A", "C")
    bn.add_edge("B", "D")
    bn.add_edge("B", "E")
    bn.add_edge("C", "F")
    bn.add_edge("C", "G")
    
    # Set parents
    bn.set_parents("A", [])
    bn.set_parents("B", ["A"])
    bn.set_parents("C", ["A"])
    bn.set_parents("D", ["B"])
    bn.set_parents("E", ["B"])
    bn.set_parents("F", ["C"])
    bn.set_parents("G", ["C"])
    
    # Random CPTs
    bn.set_cpt("A", random_binary_cpt(0, rng))
    bn.set_cpt("B", random_binary_cpt(1, rng))
    bn.set_cpt("C", random_binary_cpt(1, rng))
    bn.set_cpt("D", random_binary_cpt(1, rng))
    bn.set_cpt("E", random_binary_cpt(1, rng))
    bn.set_cpt("F", random_binary_cpt(1, rng))
    bn.set_cpt("G", random_binary_cpt(1, rng))
    
    return bn


def get_chain_7node(seed=2000):
    r"""
    7-node chain structure (matches graphs.py get_chain):
    
    A -> B -> C -> D -> E -> F -> G
    
    Linear dependency chain.
    """
    rng = np.random.default_rng(seed)
    bn = BinaryBayesNet()
    
    # Add nodes
    for n in ["A", "B", "C", "D", "E", "F", "G"]:
        bn.add_node(n)
    
    # Chain edges
    bn.add_edge("A", "B")
    bn.add_edge("B", "C")
    bn.add_edge("C", "D")
    bn.add_edge("D", "E")
    bn.add_edge("E", "F")
    bn.add_edge("F", "G")
    
    # Set parents
    bn.set_parents("A", [])
    bn.set_parents("B", ["A"])
    bn.set_parents("C", ["B"])
    bn.set_parents("D", ["C"])
    bn.set_parents("E", ["D"])
    bn.set_parents("F", ["E"])
    bn.set_parents("G", ["F"])
    
    # Random CPTs
    bn.set_cpt("A", random_binary_cpt(0, rng))
    bn.set_cpt("B", random_binary_cpt(1, rng))
    bn.set_cpt("C", random_binary_cpt(1, rng))
    bn.set_cpt("D", random_binary_cpt(1, rng))
    bn.set_cpt("E", random_binary_cpt(1, rng))
    bn.set_cpt("F", random_binary_cpt(1, rng))
    bn.set_cpt("G", random_binary_cpt(1, rng))
    
    return bn


def get_general_7node(seed=2000):
    r"""
    7-node general DAG structure built ON TOP of the 5-node general.

    The first 5 nodes (A–E) and their edges are identical to get_general_5node.
    Two new leaf nodes F and G are added with multi-parent dependencies:

    A   B          (roots, 0 parents)
    |\ /
    |  C------ (parents: A, B)
    | / \    
    D    \          (parents: A, C)
    |\    \    
    E \    |        (parents: D)          <-- same as 5-node!
    |\ \   |   
    |  F   |        (parents: D, E)
    |      |
    +------G       (parents: C, E)

    Preserved 5-node edges:  A->C, B->C, A->D, C->D, D->E
    New edges for F:         D->F, E->F
    New edges for G:         C->G, E->G
    """
    rng = np.random.default_rng(seed)
    bn = BinaryBayesNet()
    
    # Add nodes
    for n in ["A", "B", "C", "D", "E", "F", "G"]:
        bn.add_node(n)
    
    # --- Edges preserved from 5-node general ---
    bn.add_edge("A", "C")
    bn.add_edge("B", "C")
    bn.add_edge("A", "D")
    bn.add_edge("C", "D")
    bn.add_edge("D", "E")
    # --- New edges for F and G ---
    bn.add_edge("D", "F")
    bn.add_edge("E", "F")
    bn.add_edge("C", "G")
    bn.add_edge("E", "G")
    
    # Set parents (A–E identical to 5-node general)
    bn.set_parents("A", [])
    bn.set_parents("B", [])
    bn.set_parents("C", ["A", "B"])
    bn.set_parents("D", ["A", "C"])
    bn.set_parents("E", ["D"])
    bn.set_parents("F", ["D", "E"])
    bn.set_parents("G", ["C", "E"])
    
    # Random CPTs
    bn.set_cpt("A", random_binary_cpt(0, rng))
    bn.set_cpt("B", random_binary_cpt(0, rng))
    bn.set_cpt("C", random_binary_cpt(2, rng))
    bn.set_cpt("D", random_binary_cpt(2, rng))
    bn.set_cpt("E", random_binary_cpt(1, rng))
    bn.set_cpt("F", random_binary_cpt(2, rng))
    bn.set_cpt("G", random_binary_cpt(2, rng))
    
    return bn


# ============================================================
#  Convenience functions (use 7-node for training)
# ============================================================

def get_mixed_graph_structures(seed=2000):
    """
    Returns a list of 3 different 7-node graph structures for mixed training.
    
    Returns:
        list[BinaryBayesNet]: [tree, chain, general]
    """
    return [
        get_tree_7node(seed),
        get_chain_7node(seed + 1),
        get_general_7node(seed + 2),
    ]


def get_mixed_graph_structures_5node(seed=2000):
    """
    Returns a list of 3 different 5-node graph structures (legacy).
    
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
