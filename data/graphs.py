import pandas as pd
import numpy as np
from .binary_bn import BinaryBayesNet
from .categorical_bn import CategoricalBayesNet


def get_sachs_NA(seed=2000, set_random_cpts=True):
    rng = np.random.default_rng(seed)

    gt = pd.read_csv("/projects/b1094/ywl7940/JASACODE/Sachs/GroundTruth.csv")  # columns: from,to

    # IMPORTANT: node ordering must be fixed. This matches your current code.
    nodes = sorted(set(gt["from"]).union(set(gt["to"])))

    bn = BinaryBayesNet()
    for n in nodes:
        bn.add_node(n)

    # add edges
    for u, v in gt[["from", "to"]].itertuples(index=False):
        bn.add_edge(u, v)

    # parents: derive from edge list
    parents = {n: [] for n in nodes}
    for u, v in gt[["from", "to"]].itertuples(index=False):
        parents[v].append(u)

    # set parents (often required by their BN implementation, separate from add_edge)
    for n in nodes:
        # optional: keep deterministic ordering of parent lists
        bn.set_parents(n, sorted(parents[n]))

    # CPTs: only makes sense if you are *treating Sachs as binary* (it isn't originally).
    # If you only need the ground-truth structure, set_random_cpts=False.
    if set_random_cpts:
        for n in nodes:
            bn.set_cpt(n, random_binary_cpt(len(parents[n]), rng))

    return bn


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


def get_tree_10node(seed=2000):
    r"""
    10-node binary tree structure:

               A (root)
              / \
             B   C
            / \ / \
           D  E F  G
          / \
         H   I
             |
             J
    """
    rng = np.random.default_rng(seed)
    bn = BinaryBayesNet()

    for n in ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]:
        bn.add_node(n)

    bn.add_edge("A", "B")
    bn.add_edge("A", "C")
    bn.add_edge("B", "D")
    bn.add_edge("B", "E")
    bn.add_edge("C", "F")
    bn.add_edge("C", "G")
    bn.add_edge("D", "H")
    bn.add_edge("D", "I")
    bn.add_edge("I", "J")

    bn.set_parents("A", [])
    bn.set_parents("B", ["A"])
    bn.set_parents("C", ["A"])
    bn.set_parents("D", ["B"])
    bn.set_parents("E", ["B"])
    bn.set_parents("F", ["C"])
    bn.set_parents("G", ["C"])
    bn.set_parents("H", ["D"])
    bn.set_parents("I", ["D"])
    bn.set_parents("J", ["I"])

    for n in ["A"]:
        bn.set_cpt(n, random_binary_cpt(0, rng))
    for n in ["B", "C", "D", "E", "F", "G", "H", "I", "J"]:
        bn.set_cpt(n, random_binary_cpt(1, rng))

    return bn


def get_chain_10node(seed=2000):
    r"""
    10-node chain: A -> B -> C -> D -> E -> F -> G -> H -> I -> J
    """
    rng = np.random.default_rng(seed)
    bn = BinaryBayesNet()
    nodes = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

    for n in nodes:
        bn.add_node(n)

    for i in range(len(nodes) - 1):
        bn.add_edge(nodes[i], nodes[i + 1])

    bn.set_parents(nodes[0], [])
    bn.set_cpt(nodes[0], random_binary_cpt(0, rng))
    for i in range(1, len(nodes)):
        bn.set_parents(nodes[i], [nodes[i - 1]])
        bn.set_cpt(nodes[i], random_binary_cpt(1, rng))

    return bn


def get_general_10node(seed=2000):
    r"""
    10-node general DAG extending the 7-node general.

    A   B
    |\ /
    |  C       (parents: A, B)
    | / \
    D    \     (parents: A, C)
    |\    \
    E \    |   (parents: D)
    |\ \   |
    |  F   |   (parents: D, E)
    |      |
    +------G   (parents: C, E)
    
    New nodes:
      H        (parents: F, G)
      I        (parents: E, H)
      J        (parents: G, I)
    """
    rng = np.random.default_rng(seed)
    bn = BinaryBayesNet()

    for n in ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]:
        bn.add_node(n)

    # 7-node general edges
    bn.add_edge("A", "C")
    bn.add_edge("B", "C")
    bn.add_edge("A", "D")
    bn.add_edge("C", "D")
    bn.add_edge("D", "E")
    bn.add_edge("D", "F")
    bn.add_edge("E", "F")
    bn.add_edge("C", "G")
    bn.add_edge("E", "G")
    # New edges
    bn.add_edge("F", "H")
    bn.add_edge("G", "H")
    bn.add_edge("E", "I")
    bn.add_edge("H", "I")
    bn.add_edge("G", "J")
    bn.add_edge("I", "J")

    bn.set_parents("A", [])
    bn.set_parents("B", [])
    bn.set_parents("C", ["A", "B"])
    bn.set_parents("D", ["A", "C"])
    bn.set_parents("E", ["D"])
    bn.set_parents("F", ["D", "E"])
    bn.set_parents("G", ["C", "E"])
    bn.set_parents("H", ["F", "G"])
    bn.set_parents("I", ["E", "H"])
    bn.set_parents("J", ["G", "I"])

    bn.set_cpt("A", random_binary_cpt(0, rng))
    bn.set_cpt("B", random_binary_cpt(0, rng))
    bn.set_cpt("C", random_binary_cpt(2, rng))
    bn.set_cpt("D", random_binary_cpt(2, rng))
    bn.set_cpt("E", random_binary_cpt(1, rng))
    bn.set_cpt("F", random_binary_cpt(2, rng))
    bn.set_cpt("G", random_binary_cpt(2, rng))
    bn.set_cpt("H", random_binary_cpt(2, rng))
    bn.set_cpt("I", random_binary_cpt(2, rng))
    bn.set_cpt("J", random_binary_cpt(2, rng))

    return bn


def random_categorical_cpt(
    num_parents: int,
    num_categories: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Returns p[cfg, k] = P(X=k | cfg), shape (2^num_parents, num_categories), rows sum to 1."""
    if rng is None:
        rng = np.random.default_rng()
    K_configs = 1 << num_parents
    # Dirichlet(1,...,1) gives uniform over simplex; use alpha=1 per category
    p = rng.dirichlet(alpha=np.ones(num_categories), size=K_configs)
    return p.astype(np.float64)


def get_sachs(seed=2000):
    rng = np.random.default_rng(seed)
    bn = CategoricalBayesNet(cardinality=3)
    K = 3

    nodes = ["PKC", "PKA", "RAF", "Mek", "Erk", "Akt",
             "Jnk", "P38", "Plcg", "PIP3", "PIP2"]
    for n in nodes:
        bn.add_node(n)

    # Edges
    bn.add_edge("PKC", "Jnk")
    bn.add_edge("PKC", "P38")
    bn.add_edge("PKC", "PKA")
    bn.add_edge("PKC", "RAF")
    bn.add_edge("PKC", "Mek")
    bn.add_edge("PKA", "Jnk")
    bn.add_edge("PKA", "P38")
    bn.add_edge("PKA", "RAF")
    bn.add_edge("PKA", "Mek")
    bn.add_edge("PKA", "Erk")
    bn.add_edge("PKA", "Akt")
    bn.add_edge("RAF", "Mek")
    bn.add_edge("Mek", "Erk")
    bn.add_edge("Erk", "Akt")
    bn.add_edge("Plcg", "PIP2")
    bn.add_edge("Plcg", "PIP3")
    bn.add_edge("PIP3", "PIP2")
    bn.add_edge("PIP3", "Akt")

    # Parents
    bn.set_parents("PKC",  [])
    bn.set_parents("Plcg", [])
    bn.set_parents("PKA",  ["PKC"])
    bn.set_parents("RAF",  ["PKC", "PKA"])
    bn.set_parents("Jnk",  ["PKC", "PKA"])
    bn.set_parents("P38",  ["PKC", "PKA"])
    bn.set_parents("Mek",  ["PKC", "PKA", "RAF"])
    bn.set_parents("Erk",  ["PKA", "Mek"])
    bn.set_parents("Akt",  ["PKA", "Erk", "PIP3"])
    bn.set_parents("PIP3", ["Plcg"])
    bn.set_parents("PIP2", ["Plcg", "PIP3"])

    # CPTs — num_parents determines table size
    bn.set_cpt("PKC",  random_categorical_cpt_K(0, K, rng))  # root
    bn.set_cpt("Plcg", random_categorical_cpt_K(0, K, rng))  # root
    bn.set_cpt("PKA",  random_categorical_cpt_K(1, K, rng))  # 1 parent
    bn.set_cpt("RAF",  random_categorical_cpt_K(2, K, rng))  # 2 parents
    bn.set_cpt("Jnk",  random_categorical_cpt_K(2, K, rng))  # 2 parents
    bn.set_cpt("P38",  random_categorical_cpt_K(2, K, rng))  # 2 parents
    bn.set_cpt("Mek",  random_categorical_cpt_K(3, K, rng))  # 3 parents
    bn.set_cpt("Erk",  random_categorical_cpt_K(2, K, rng))  # 2 parents
    bn.set_cpt("Akt",  random_categorical_cpt_K(3, K, rng))  # 3 parents
    bn.set_cpt("PIP3", random_categorical_cpt_K(1, K, rng))  # 1 parent
    bn.set_cpt("PIP2", random_categorical_cpt_K(2, K, rng))  # 2 parents

    return bn


def random_categorical_cpt_3(
    num_parents: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Returns p[cfg, k] = P(X=k | cfg), shape (3^num_parents, 3), rows sum to 1.
    For use with CategoricalBayesNet(cardinality=3); parent configs are 3^k.
    """
    if rng is None:
        rng = np.random.default_rng()
    K_configs = 3 ** num_parents
    p = rng.dirichlet(alpha=np.ones(3), size=K_configs)
    return p.astype(np.float64)


def random_categorical_cpt_K(
    num_parents: int,
    num_classes: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Returns p[cfg, k] = P(X=k | cfg), shape (num_classes^num_parents, num_classes), rows sum to 1.
    For use with CategoricalBayesNet(cardinality=num_classes).
    """
    if rng is None:
        rng = np.random.default_rng()
    K_configs = num_classes ** num_parents
    p = rng.dirichlet(alpha=np.ones(num_classes), size=K_configs)
    return p.astype(np.float64)


def get_sachs_categorical(seed=2000, set_random_cpts=True, num_classes=3):
    """
    Same DAG as get_sachs (Sachs ground truth) but with a categorical BN (num_classes per node).
    Returns CategoricalBayesNet(cardinality=num_classes). Default num_classes=3.
    """
    rng = np.random.default_rng(seed)
    gt = pd.read_csv("/projects/b1094/ywl7940/JASACODE/Sachs/GroundTruth.csv")
    nodes = sorted(set(gt["from"]).union(set(gt["to"])))

    bn = CategoricalBayesNet(cardinality=num_classes)
    for n in nodes:
        bn.add_node(n)
    for u, v in gt[["from", "to"]].itertuples(index=False):
        bn.add_edge(u, v)

    parents = {n: [] for n in nodes}
    for u, v in gt[["from", "to"]].itertuples(index=False):
        parents[v].append(u)
    for n in nodes:
        bn.set_parents(n, sorted(parents[n]))

    if set_random_cpts:
        for n in nodes:
            bn.set_cpt(n, random_categorical_cpt_K(len(parents[n]), num_classes, rng))
    else:
        # Use deterministic uniform CPTs purely to satisfy the compile-time
        # requirement that every node has a CPT. Downstream code (e.g.,
        # eval_sachs_real.py) replaces these CPTs with empirical ones and
        # only relies on the structure extracted from the compiled template.
        for n in nodes:
            num_parents = len(parents[n])
            num_configs = num_classes ** num_parents
            uniform_row = np.full((num_configs, num_classes), 1.0 / num_classes, dtype=np.float64)
            bn.set_cpt(n, uniform_row)

    return bn


def get_general(seed=2000):

    rng = np.random.default_rng(seed)
    bn = BinaryBayesNet()
    for n in ["A", "B", "C", "D", "E"]:
        bn.add_node(n)

    bn.add_edge("A", "C")
    bn.add_edge("B", "C")

    bn.add_edge("C", "D")
    bn.add_edge("A", "D")
    bn.add_edge("D", "E")

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


def get_chain(seed=2000):

    rng = np.random.default_rng(seed)
    bn = BinaryBayesNet()
    for n in ["A", "B", "C", "D", "E", "F", "G"]:
        bn.add_node(n)

    bn.add_edge("A", "B")
    bn.add_edge("B", "C")
    bn.add_edge("C", "D")
    bn.add_edge("D", "E")
    bn.add_edge("E", "F")
    bn.add_edge("F", "G")

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


def get_tree(seed=2000):

    rng = np.random.default_rng(seed)
    bn = BinaryBayesNet()
    for n in ["A", "B", "C", "D", "E", "F", "G"]:
        bn.add_node(n)

    bn.add_edge("A", "B")
    bn.add_edge("A", "C")

    bn.add_edge("B", "D")
    bn.add_edge("B", "E")

    bn.add_edge("C", "F")
    bn.add_edge("C", "G")

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



def get_general5(seed=2000):

    rng = np.random.default_rng(seed)
    bn = BinaryBayesNet()
    for n in ["A", "B", "C", "D", "E"]:
        bn.add_node(n)

    bn.add_edge("A", "C")
    bn.add_edge("B", "C")

    bn.add_edge("C", "D")
    bn.add_edge("A", "D")
    bn.add_edge("D", "E")

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


def get_chain5(seed=2000):

    rng = np.random.default_rng(seed)
    bn = BinaryBayesNet()
    for n in ["A", "B", "C", "D", "E"]:
        bn.add_node(n)

    bn.add_edge("A", "B")
    bn.add_edge("B", "C")
    bn.add_edge("C", "D")
    bn.add_edge("D", "E")
    # bn.add_edge("E", "F")
    # bn.add_edge("F", "G")

    bn.set_parents("A", [])
    bn.set_parents("B", ["A"])
    bn.set_parents("C", ["B"])
    bn.set_parents("D", ["C"])
    bn.set_parents("E", ["D"])
    # bn.set_parents("F", ["E"])
    # bn.set_parents("G", ["F"])

    # Random CPTs
    bn.set_cpt("A", random_binary_cpt(0, rng))
    bn.set_cpt("B", random_binary_cpt(1, rng))
    bn.set_cpt("C", random_binary_cpt(1, rng))
    bn.set_cpt("D", random_binary_cpt(1, rng))
    bn.set_cpt("E", random_binary_cpt(1, rng))
    # bn.set_cpt("F", random_binary_cpt(1, rng))
    # bn.set_cpt("G", random_binary_cpt(1, rng))

    return bn


def get_tree5(seed=2000):

    rng = np.random.default_rng(seed)
    bn = BinaryBayesNet()
    for n in ["A", "B", "C", "D", "E"]:
        bn.add_node(n)

    bn.add_edge("A", "B")
    bn.add_edge("A", "C")

    bn.add_edge("B", "D")
    bn.add_edge("B", "E")

    # bn.add_edge("C", "F")
    # bn.add_edge("C", "G")

    bn.set_parents("A", [])
    bn.set_parents("B", ["A"])
    bn.set_parents("C", ["A"])
    bn.set_parents("D", ["B"])
    bn.set_parents("E", ["B"])
    # bn.set_parents("F", ["C"])
    # bn.set_parents("G", ["C"])

    # Random CPTs
    bn.set_cpt("A", random_binary_cpt(0, rng))
    bn.set_cpt("B", random_binary_cpt(1, rng))
    bn.set_cpt("C", random_binary_cpt(1, rng))
    bn.set_cpt("D", random_binary_cpt(1, rng))
    bn.set_cpt("E", random_binary_cpt(1, rng))
    # bn.set_cpt("F", random_binary_cpt(1, rng))
    # bn.set_cpt("G", random_binary_cpt(1, rng))

    return bn



def get_general7(seed=2000):
    r"""
    7-node general DAG structure built ON TOP of the 5-node general.

    The first 5 nodes (A–E) and their edges are identical to get_general_5node.
    Two new leaf nodes F and G are added with multi-parent dependencies:

    A   B          (roots, 0 parents)
    |\ /
    | C------+     (parents: A, B)
    | / \    |
    D    |   |     (parents: A, C)
    |\   |   |
    E \  |   |     (parents: D)          <-- same as 5-node!
    |\ \ |   |
    | F  |   |     (parents: D, E)
    |     \ /
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