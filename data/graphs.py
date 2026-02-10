import pandas as pd
import numpy as np
from .binary_bn import BinaryBayesNet

def get_sachs(seed=2000, set_random_cpts=True):
    rng = np.random.default_rng(seed)

    gt = pd.read_csv("/home/dennis/JASACODE/Sachs/GroundTruth.csv")  # columns: from,to

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