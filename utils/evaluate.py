from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Dict, Any, Optional

import numpy as np
import torch

from bn_template import BNTemplate
from multigraph_sampler import sample_many_graphs
from binary_bn import BNError


@dataclass
class EvalSpec:
    # context lengths to test (m). Sequence length is L=m+1.
    context_lens: Sequence[int]
    # how many independent episodes per (target, context_len)
    num_episodes: int = 2000
    # RNG seed for evaluation sampling
    seed: int = 0
    # output CSV path
    output_csv: str = "eval_tv.csv"
    # device for model inference
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    # optionally cap batch size for inference to avoid GPU OOM
    infer_batch_size: int = 512


def _compute_parent_cfg(
    test_prefix: np.ndarray,          # shape (B, N), values in {0,1}
    parent_idx: np.ndarray,           # shape (k,)
) -> np.ndarray:
    """
    parent_config id = sum parent_value[j] * 2^j in the *given parent order*.
    Returns shape (B,) int64.
    """
    k = int(parent_idx.size)
    if k == 0:
        return np.zeros((test_prefix.shape[0],), dtype=np.int64)
    pv = test_prefix[:, parent_idx].astype(np.int64)  # (B,k)
    weights = (1 << np.arange(k, dtype=np.int64))[None, :]  # (1,k)
    return (pv * weights).sum(axis=1)


def _build_icl_x(
    X_full: np.ndarray,      # (B, L, N) uint8 full samples
    target_index: int,       # t
) -> np.ndarray:
    """
    Build x: (B, L, N+1) int64 according to your masking rules:
      - context rows (0..L-2): mask t+1..N-1, keep target t
      - test row (L-1): mask t..N-1 (target included)
      - append final feature = target_index (same for all rows)
    """
    B, L, N = X_full.shape
    t = int(target_index)
    X_mask = X_full.copy()

    # context rows: mask future nodes only
    if t + 1 < N:
        X_mask[:, : L - 1, t + 1 :] = 0

    # test row: mask target and future
    X_mask[:, L - 1, t:] = 0

    tgt_feat = np.full((B, L, 1), t, dtype=np.int64)
    X_out = np.concatenate([X_mask.astype(np.int64), tgt_feat], axis=2)
    return X_out  # (B, L, N+1)


def evaluate_tv_over_context(
    model: torch.nn.Module,
    template: BNTemplate,
    p1_list_fixed: List[np.ndarray],
    spec: EvalSpec,
) -> None:
    """
    Writes a CSV with per-episode predictions and ground truth.

    p1_list_fixed: length N; each entry shape (1, 2^k_i) for the SINGLE fixed BN.
    """
    # Basic checks
    N = template.num_nodes
    if len(p1_list_fixed) != N:
        raise BNError("p1_list_fixed must have length equal to template.num_nodes")
    for i, parents in enumerate(template.parent_idx):
        k = int(parents.size)
        K = 1 << k
        if p1_list_fixed[i].shape != (1, K):
            raise BNError(f"p1_list_fixed[{i}] must have shape (1,{K}), got {p1_list_fixed[i].shape}")

    rng = np.random.default_rng(spec.seed)

    device = torch.device(spec.device)
    model = model.to(device)
    model.eval()

    out_path = Path(spec.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # CSV schema: one row per episode
    fieldnames = [
        "context_len",
        "target_index",
        "episode",
        "p_hat",
        "p_true",
        "tv",
        "y_test",
        "parents_cfg",
    ]

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        # Evaluate for each context length
        for m in spec.context_lens:
            L = int(m) + 1

            # We'll batch episodes for efficient inference
            remaining = int(spec.num_episodes)
            episode_offset = 0

            while remaining > 0:
                B = min(remaining, spec.infer_batch_size)

                # Sample B episodes from the single fixed BN in parallel:
                # graph_ids are all zeros because p1_list_fixed has 1 graph
                graph_ids = np.zeros((B,), dtype=np.int64)

                X_full = sample_many_graphs(
                    template=template,
                    p1_list=p1_list_fixed,
                    graph_ids=graph_ids,
                    num_examples=L,
                    rng=rng,
                )  # (B, L, N)

                t = None  # assigned per loop below

                # For each target node, run the model and compute TV
                for t in range(N):
                    # Model input x
                    X_out = _build_icl_x(X_full, target_index=t)  # (B, L, N+1)

                    # True label y_test
                    y_test = X_full[:, L - 1, t].astype(np.int64)  # (B,)

                    # Ground-truth conditional p_true from the BN CPT using parents in the TEST token.
                    # Evidence available in test token is nodes < t (since t.. masked).
                    # The BN conditional itself depends only on parents(t), which are among earlier nodes in topo order.
                    test_prefix = X_full[:, L - 1, :]  # (B, N) full values (use as ground truth for parent config)
                    parents_idx = template.parent_idx[t]
                    cfg = _compute_parent_cfg(test_prefix, parents_idx)  # (B,)
                    p_true = p1_list_fixed[t][0, cfg]  # (B,) float64

                    # Model prediction p_hat
                    x_tensor = torch.as_tensor(X_out, dtype=torch.float32, device=device)  # float for read_in
                    with torch.no_grad():
                        logits = model(x_tensor)  # expected (B,)
                        p_hat = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float64)

                    tv = np.abs(p_hat - p_true)

                    # Write per-episode rows
                    for i in range(B):
                        writer.writerow(
                            {
                                "context_len": int(m),
                                "target_index": int(t),
                                "episode": int(episode_offset + i),
                                "p_hat": float(p_hat[i]),
                                "p_true": float(p_true[i]),
                                "tv": float(tv[i]),
                                "y_test": int(y_test[i]),
                                "parents_cfg": int(cfg[i]),
                            }
                        )

                episode_offset += B
                remaining -= B
