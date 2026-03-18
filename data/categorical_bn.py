# categorical_bn.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Optional
import numpy as np

from .binary_bn import _topological_sort, BNError


@dataclass(frozen=True)
class CategoricalNodeSpec:
    """
    Sampling-ready node spec for categorical BN.
    parents: (k,) parent node indices in topo order.
    cardinality: K (number of values 0..K-1).
    cpt: (K^k, K) float64, cpt[cfg, v] = P(X=v | parent_config=cfg).
        Parent config: cfg = sum_{j} parent_val[j] * K^j (parents in order).
    """
    parents: np.ndarray   # (k,)
    cardinality: int      # K
    cpt: np.ndarray      # (K^k, K), rows sum to 1


class CategoricalBayesNet:
    """
    Bayesian network where each node takes values in {0, 1, ..., K-1}.
    All nodes share the same cardinality K (fixed for the whole network).
    """

    def __init__(self, cardinality: int = 3) -> None:
        if cardinality < 2:
            raise BNError("cardinality must be >= 2 (use binary_bn for K=2).")
        self._K = int(cardinality)
        self._nodes: List[str] = []
        self._node_set: set[str] = set()
        self._parents: Dict[str, List[str]] = {}
        self._cpt: Dict[str, np.ndarray] = {}  # node -> (num_configs, K)
        self._frozen: bool = False

    @property
    def nodes(self) -> List[str]:
        return list(self._nodes)

    @property
    def cardinality(self) -> int:
        return self._K

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
        if parent == child or parent in self._parents[child]:
            return
        self._parents[child].append(parent)

    def set_parents(self, node: str, parents_ordered: Sequence[str]) -> None:
        if self._frozen:
            raise BNError("Cannot modify: network is frozen/compiled.")
        if node not in self._node_set:
            raise BNError(f"Unknown node '{node}'.")
        for p in parents_ordered:
            if p not in self._node_set or p == node:
                raise BNError(f"Invalid parent '{p}'.")
        self._parents[node] = list(parents_ordered)

    def set_cpt(self, node: str, cpt: np.ndarray, parents_ordered: Optional[Sequence[str]] = None) -> None:
        """
        cpt: (num_configs, K) with num_configs = K^k (k = number of parents).
        Each row must sum to 1. Values in [0,1].
        """
        if self._frozen:
            raise BNError("Cannot modify: network is frozen/compiled.")
        if node not in self._node_set:
            raise BNError(f"Unknown node '{node}'.")
        if parents_ordered is not None:
            self.set_parents(node, parents_ordered)

        k = len(self._parents[node])
        expected_configs = self._K ** k
        cpt = np.asarray(cpt, dtype=np.float64)
        if cpt.ndim != 2:
            raise BNError(f"CPT for '{node}' must be 2D (num_configs, K).")
        if cpt.shape[0] != expected_configs or cpt.shape[1] != self._K:
            raise BNError(f"CPT for '{node}' must have shape ({expected_configs}, {self._K}), got {cpt.shape}.")
        if np.any(cpt < 0) or np.any(cpt > 1):
            raise BNError(f"CPT for '{node}' has entries outside [0,1].")
        row_sums = cpt.sum(axis=1)
        if not np.allclose(row_sums, 1.0):
            raise BNError(f"CPT for '{node}' rows must sum to 1.")
        self._cpt[node] = cpt.copy()

    def get_cpt(self, node: str) -> tuple[List[str], np.ndarray]:
        if node not in self._node_set or node not in self._cpt:
            raise BNError(f"No CPT set for node '{node}'.")
        return list(self._parents[node]), self._cpt[node].copy()

    def get_parents(self, node: str) -> List[str]:
        if node not in self._node_set:
            raise BNError(f"Unknown node '{node}'.")
        return list(self._parents[node])

    def _validate_all_cpts_set(self) -> None:
        missing = [n for n in self._nodes if n not in self._cpt]
        if missing:
            raise BNError(f"Missing CPTs for nodes: {missing}")

    def compile(self) -> "CompiledCategoricalBayesNet":
        self._validate_all_cpts_set()
        topo = _topological_sort(self._nodes, self._parents)
        name_to_idx = {n: i for i, n in enumerate(topo)}
        specs: List[CategoricalNodeSpec] = []
        for n in topo:
            ps = self._parents[n]
            parent_idx = np.array([name_to_idx[p] for p in ps], dtype=np.int64)
            cpt = np.asarray(self._cpt[n], dtype=np.float64)
            specs.append(CategoricalNodeSpec(parents=parent_idx, cardinality=self._K, cpt=cpt))
        self._frozen = True
        return CompiledCategoricalBayesNet(
            topo_nodes=topo,
            node_to_index=name_to_idx,
            specs=specs,
            cardinality=self._K,
        )


@dataclass(frozen=True)
class CompiledCategoricalBayesNet:
    topo_nodes: List[str]
    node_to_index: Dict[str, int]
    specs: List[CategoricalNodeSpec]
    cardinality: int

    @property
    def num_nodes(self) -> int:
        return len(self.topo_nodes)
