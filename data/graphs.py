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