# binary_bn.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple, Optional
import numpy as np


class BNError(RuntimeError):
    pass


def _topological_sort(nodes: Sequence[str], parents: Dict[str, List[str]]) -> List[str]:
    """
    Kahn's algorithm. Raises BNError if a cycle is detected.
    """
    indeg = {n: 0 for n in nodes}
    children: Dict[str, List[str]] = {n: [] for n in nodes}

    for child in nodes:
        for p in parents.get(child, []):
            if p not in indeg:
                raise BNError(f"Parent '{p}' not found among nodes.")
            indeg[child] += 1
            children[p].append(child)

    q = [n for n in nodes if indeg[n] == 0]
    out: List[str] = []

    while q:
        n = q.pop()
        out.append(n)
        for c in children[n]:
            indeg[c] -= 1
            if indeg[c] == 0:
                q.append(c)

    if len(out) != len(nodes):
        raise BNError("Cycle detected in DAG (topological sort failed).")
    return out


@dataclass(frozen=True)
class NodeSpec:
    """
    Sampling-ready node specification.

    parents: ordered list of parent node indices.
    p1: vector of length 2^k giving P(X=1 | parent_config).
        parent_config is encoded as bitmask over parents in this exact order:
            cfg = sum_{j=0..k-1} parent_value[j] * 2^j
    """
    parents: np.ndarray  # shape (k,), dtype=int64
    p1: np.ndarray       # shape (2^k,), dtype=float64


class BinaryBayesNet:
    """
    Binary Bayesian Network with explicit CPTs.

    - Nodes are named strings (you can use ints too, but keep consistent).
    - All variables are binary in {0,1}.
    - CPT for node i with k parents is represented as p1[cfg] = P(X_i=1 | cfg).
    """

    def __init__(self) -> None:
        self._nodes: List[str] = []
        self._node_set: set[str] = set()
        self._parents: Dict[str, List[str]] = {}
        self._cpt_p1: Dict[str, np.ndarray] = {}  # node -> (2^k,) float
        self._frozen: bool = False

    @property
    def nodes(self) -> List[str]:
        return list(self._nodes)

    def add_node(self, name: str) -> None:
        if self._frozen:
            raise BNError("Cannot modify: network is frozen/compiled.")
        if name in self._node_set:
            return
        self._nodes.append(name)
        self._node_set.add(name)
        self._parents[name] = []

    def add_edge(self, parent: str, child: str) -> None:
        if self._frozen:
            raise BNError("Cannot modify: network is frozen/compiled.")
        if parent not in self._node_set or child not in self._node_set:
            raise BNError("Both parent and child must be added as nodes before adding an edge.")
        if parent == child:
            raise BNError("Self-loops are not allowed.")
        if parent in self._parents[child]:
            return
        self._parents[child].append(parent)

    def set_parents(self, node: str, parents_ordered: Sequence[str]) -> None:
        """
        Set (and order) parents explicitly. This order defines the CPT indexing.
        """
        if self._frozen:
            raise BNError("Cannot modify: network is frozen/compiled.")
        if node not in self._node_set:
            raise BNError(f"Unknown node '{node}'.")
        for p in parents_ordered:
            if p not in self._node_set:
                raise BNError(f"Unknown parent '{p}'.")
            if p == node:
                raise BNError("Self parent is not allowed.")
        self._parents[node] = list(parents_ordered)

    def set_cpt(self, node: str, p1: np.ndarray, parents_ordered: Optional[Sequence[str]] = None) -> None:
        """
        Provide p1[cfg] = P(node=1 | cfg). Must have length 2^k where k=len(parents).
        If parents_ordered is provided, it sets the parent order first.
        """
        if self._frozen:
            raise BNError("Cannot modify: network is frozen/compiled.")
        if node not in self._node_set:
            raise BNError(f"Unknown node '{node}'.")

        if parents_ordered is not None:
            self.set_parents(node, parents_ordered)

        k = len(self._parents[node])
        p1 = np.asarray(p1, dtype=np.float64).reshape(-1)

        expected = 1 << k
        if p1.shape[0] != expected:
            raise BNError(f"CPT for '{node}' must have length 2^{k}={expected}, got {p1.shape[0]}.")

        if np.any(p1 < 0.0) or np.any(p1 > 1.0):
            raise BNError(f"CPT for '{node}' has probabilities outside [0,1].")

        self._cpt_p1[node] = p1.copy()

    def get_cpt(self, node: str) -> Tuple[List[str], np.ndarray]:
        """
        Returns (parents_ordered, p1_vector).
        """
        if node not in self._node_set:
            raise BNError(f"Unknown node '{node}'.")
        if node not in self._cpt_p1:
            raise BNError(f"No CPT set for node '{node}'.")
        return list(self._parents[node]), self._cpt_p1[node].copy()

    def _validate_all_cpts_set(self) -> None:
        missing = [n for n in self._nodes if n not in self._cpt_p1]
        if missing:
            raise BNError(f"Missing CPTs for nodes: {missing}")

    def _validate_dag(self) -> List[str]:
        topo = _topological_sort(self._nodes, self._parents)
        return topo

    def compile(self) -> "CompiledBinaryBayesNet":
        """
        Freeze BN and produce a compiled, sampling-ready representation.
        """
        self._validate_all_cpts_set()
        topo = self._validate_dag()

        name_to_idx = {n: i for i, n in enumerate(topo)}
        specs: List[NodeSpec] = []

        for n in topo:
            ps = self._parents[n]
            parent_idx = np.array([name_to_idx[p] for p in ps], dtype=np.int64)
            p1 = np.asarray(self._cpt_p1[n], dtype=np.float64).reshape(-1)
            specs.append(NodeSpec(parents=parent_idx, p1=p1))

        self._frozen = True
        return CompiledBinaryBayesNet(
            topo_nodes=topo,
            node_to_index=name_to_idx,
            specs=specs,
        )


@dataclass(frozen=True)
class CompiledBinaryBayesNet:
    topo_nodes: List[str]
    node_to_index: Dict[str, int]
    specs: List[NodeSpec]

    @property
    def num_nodes(self) -> int:
        return len(self.topo_nodes)


# In binary_bn.py (inside class BinaryBayesNet)

def get_parents(self, node: str) -> list[str]:
    if node not in self._node_set:
        raise BNError(f"Unknown node '{node}'.")
    return list(self._parents[node])
