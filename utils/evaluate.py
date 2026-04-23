from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Optional

import numpy as np
import torch

from data.bn_template import BNTemplate
from data.multigraph_sampler import sample_many_graphs
from data.binary_bn import BNError


@dataclass
class EvalSpec:
    context_lens: Sequence[int]
    num_episodes: int = 2000
    seed: int = 0
    output_csv: str = "eval_tv.csv"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    infer_batch_size: int = 512


def _compute_parent_cfg(
    test_prefix: np.ndarray,
    parent_idx: np.ndarray,
) -> np.ndarray:
    """parent_config id = sum parent_value[j] * 2^j. Returns (B,) int64."""
    k = int(parent_idx.size)
    if k == 0:
        return np.zeros((test_prefix.shape[0],), dtype=np.int64)
    pv = test_prefix[:, parent_idx].astype(np.int64)
    weights = (1 << np.arange(k, dtype=np.int64))[None, :]
    return (pv * weights).sum(axis=1)


def _compute_prefix_cfg_binary(rows: np.ndarray) -> np.ndarray:
    """Config id for prefix (x_0..x_{t-1}); rows (B, nprev), binary digits."""
    nprev = int(rows.shape[1])
    weights = (1 << np.arange(nprev, dtype=np.int64))[None, :]
    return (rows.astype(np.int64) * weights).sum(axis=1)


def _icl_masked_x(X_full: np.ndarray, target_index: int) -> np.ndarray:
    """Same node masking as the model ICL input (without appended target_index channel).

    Context rows: zero columns t+1..N-1. Test row: zero columns t..N-1.
    """
    B, L, N = X_full.shape
    t = int(target_index)
    X_mask = X_full.copy()
    if t + 1 < N:
        X_mask[:, : L - 1, t + 1 :] = 0
    X_mask[:, L - 1, t:] = 0
    return X_mask.astype(np.int64, copy=False)


def _build_icl_x(
    X_full: np.ndarray,
    target_index: int,
) -> np.ndarray:
    """Build model input (B, L, N+1) with masking and appended target_index."""
    B, L, N = X_full.shape
    t = int(target_index)
    X_mask = _icl_masked_x(X_full, target_index)
    tgt_feat = np.full((B, L, 1), t, dtype=np.int64)
    X_out = np.concatenate([X_mask, tgt_feat], axis=2)
    return X_out


def _tv_abs_binary_safe(p_hat: np.ndarray, p_true: np.ndarray) -> np.ndarray:
    """|p_hat - p_true|; non-finite -> 1.0."""
    tv = np.abs(p_hat - p_true)
    return np.where(np.isfinite(tv), tv, 1.0)


def _naive_graph_agnostic_from_context(
    X_full: np.ndarray,
    target_index: int,
) -> np.ndarray:
    """
    Paper-correct naive: MLE P(X_t=1 | x_0..x_{t-1}) from masked context.
    Conditions on ALL preceding variables (= fully-connected MLE, ignores true DAG).
    t==0 -> marginal over context. Zero matching context rows -> nan.
    """
    B, L, N = X_full.shape
    m = L - 1
    t = int(target_index)
    X_mask = _icl_masked_x(X_full, target_index)
    X_ctx = X_mask[:, :m, :]
    y_ctx = X_ctx[:, :, t].astype(np.int64)
    nprev = t
    Kconfigs = 1 << nprev

    if nprev == 0:
        cfg_ctx = np.zeros((B, m), dtype=np.int64)
        cfg_test = np.zeros((B,), dtype=np.int64)
    else:
        flat = X_ctx[:, :, :nprev].reshape(B * m, nprev)
        cfg_ctx = _compute_prefix_cfg_binary(flat).reshape(B, m)
        cfg_test = _compute_prefix_cfg_binary(X_mask[:, L - 1, :nprev])

    p_hat = np.empty((B,), dtype=np.float64)
    for i in range(B):
        tot = np.bincount(cfg_ctx[i], minlength=Kconfigs).astype(np.float64)
        one = np.bincount(cfg_ctx[i], weights=y_ctx[i].astype(np.float64), minlength=Kconfigs)
        c = int(cfg_test[i])
        den = tot[c]
        p_hat[i] = one[c] / den if den > 0 else np.nan
    return p_hat


def _naive_context_tot_at_prefix(
    X_full: np.ndarray,
    target_index: int,
) -> np.ndarray:
    """Count context rows whose masked prefix matches test prefix. (B,) int64."""
    B, L, N = X_full.shape
    m = L - 1
    t = int(target_index)
    nprev = t
    Kconfigs = 1 << nprev
    X_mask = _icl_masked_x(X_full, target_index)
    X_ctx = X_mask[:, :m, :]

    if nprev == 0:
        cfg_ctx = np.zeros((B, m), dtype=np.int64)
        cfg_test = np.zeros((B,), dtype=np.int64)
    else:
        flat = X_ctx[:, :, :nprev].reshape(B * m, nprev)
        cfg_ctx = _compute_prefix_cfg_binary(flat).reshape(B, m)
        cfg_test = _compute_prefix_cfg_binary(X_mask[:, L - 1, :nprev])

    out = np.empty((B,), dtype=np.int64)
    for i in range(B):
        tot = np.bincount(cfg_ctx[i], minlength=Kconfigs).astype(np.int64)
        out[i] = tot[int(cfg_test[i])]
    return out


def _bayes_cpt_from_context(
    X_full: np.ndarray,
    template: BNTemplate,
    target_index: int,
) -> np.ndarray:
    """
    Bayesian CPT baseline (known DAG): MLE P(X_t=1 | parent_cfg) from masked context.
    Zero count for test parent config -> nan.
    """
    B, L, N = X_full.shape
    m = L - 1
    t = int(target_index)
    parents_idx = template.parent_idx[t]
    k = int(parents_idx.size)
    K = 1 << k

    X_mask = _icl_masked_x(X_full, target_index)
    X_ctx = X_mask[:, :m, :]
    y_ctx = X_ctx[:, :, t].astype(np.int64)

    if k == 0:
        cfg_ctx = np.zeros((B, m), dtype=np.int64)
        cfg_test = np.zeros((B,), dtype=np.int64)
    else:
        cfg_ctx = _compute_parent_cfg(
            X_ctx.reshape(B * m, N), parents_idx
        ).reshape(B, m)
        cfg_test = _compute_parent_cfg(X_mask[:, L - 1, :], parents_idx)

    p_hat = np.empty((B,), dtype=np.float64)
    for i in range(B):
        tot = np.bincount(cfg_ctx[i], minlength=K).astype(np.float64)
        one = np.bincount(cfg_ctx[i], weights=y_ctx[i], minlength=K).astype(np.float64)
        cfg = int(cfg_test[i])
        den = tot[cfg]
        p_hat[i] = one[cfg] / den if den > 0 else np.nan
    return p_hat


def _bayes_context_tot_at_test_cfg(
    X_full: np.ndarray,
    template: BNTemplate,
    target_index: int,
) -> np.ndarray:
    """Count context rows whose parent cfg matches test parent cfg. (B,) int64."""
    B, L, N = X_full.shape
    m = L - 1
    t = int(target_index)
    parents_idx = template.parent_idx[t]
    k = int(parents_idx.size)
    K = 1 << k

    X_mask = _icl_masked_x(X_full, target_index)
    X_ctx = X_mask[:, :m, :]
    if k == 0:
        cfg_ctx = np.zeros((B, m), dtype=np.int64)
        cfg_test = np.zeros((B,), dtype=np.int64)
    else:
        cfg_ctx = _compute_parent_cfg(
            X_ctx.reshape(B * m, N), parents_idx
        ).reshape(B, m)
        cfg_test = _compute_parent_cfg(X_mask[:, L - 1, :], parents_idx)

    out = np.empty((B,), dtype=np.int64)
    for i in range(B):
        tot = np.bincount(cfg_ctx[i], minlength=K).astype(np.int64)
        out[i] = tot[int(cfg_test[i])]
    return out


def evaluate_tv_over_context_with_baselines(
    model: torch.nn.Module,
    template: BNTemplate,
    p1_list_fixed: List[np.ndarray],
    spec: EvalSpec,
) -> None:
    """
    Evaluates model, naive baseline (paper-correct: FC MLE), and Bayesian baseline (known DAG).

    Naive: graph-agnostic MLE P(X_t | x_0..x_{t-1}) — conditions on all preceding nodes.
    Bayes: MLE using true parents only. No smoothing; non-finite TV -> 1.0.
    If m=0, tv_naive is 1.0. If no context row matches the relevant cfg, TV -> 1.0.
    """
    N = template.num_nodes
    if len(p1_list_fixed) != N:
        raise BNError("p1_list_fixed must have length equal to template.num_nodes")
    num_graphs = p1_list_fixed[0].shape[0]
    for i, parents in enumerate(template.parent_idx):
        k = int(parents.size)
        K = 1 << k
        if p1_list_fixed[i].shape != (num_graphs, K):
            raise BNError(f"p1_list_fixed[{i}] must have shape ({num_graphs},{K}), got {p1_list_fixed[i].shape}")

    rng = np.random.default_rng(spec.seed)

    device = torch.device(spec.device)
    model = model.to(device)
    model.eval()

    out_path = Path(spec.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "context_len",
        "target_index",
        "episode",
        "p_true",
        "y_test",
        "parents_cfg",
        "p_hat_model",
        "tv_model",
        "p_hat_naive",
        "tv_naive",
        "p_hat_bayes",
        "tv_bayes",
    ]

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for m in spec.context_lens:
            L = int(m) + 1
            remaining = int(spec.num_episodes)
            episode_offset = 0

            while remaining > 0:
                B = min(remaining, spec.infer_batch_size)

                graph_ids = rng.integers(0, num_graphs, size=B, dtype=np.int64)
                X_full = sample_many_graphs(
                    template=template,
                    p1_list=p1_list_fixed,
                    graph_ids=graph_ids,
                    num_examples=L,
                    rng=rng,
                )

                for t in range(N):
                    y_test = X_full[:, L - 1, t].astype(np.int64)

                    test_prefix = X_full[:, L - 1, :]
                    parents_idx = template.parent_idx[t]
                    cfg = _compute_parent_cfg(test_prefix, parents_idx)
                    p_true = p1_list_fixed[t][graph_ids, cfg].astype(np.float64)

                    X_out = _build_icl_x(X_full, target_index=t)
                    x_tensor = torch.as_tensor(X_out, dtype=torch.float32, device=device)
                    with torch.no_grad():
                        logits = model(x_tensor)
                        p_hat_model = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float64)
                    tv_model = _tv_abs_binary_safe(p_hat_model, p_true)

                    p_hat_naive = _naive_graph_agnostic_from_context(X_full, t).astype(np.float64)
                    tv_naive = _tv_abs_binary_safe(p_hat_naive, p_true)
                    if m == 0:
                        tv_naive = np.ones_like(tv_naive)
                    naive_tot = _naive_context_tot_at_prefix(X_full, t)
                    tv_naive = np.where(naive_tot == 0, 1.0, tv_naive)

                    p_hat_bayes = _bayes_cpt_from_context(X_full, template, t).astype(np.float64)
                    tv_bayes = _tv_abs_binary_safe(p_hat_bayes, p_true)
                    bayes_tot = _bayes_context_tot_at_test_cfg(X_full, template, t)
                    tv_bayes = np.where(bayes_tot == 0, 1.0, tv_bayes)

                    for i in range(B):
                        writer.writerow(
                            {
                                "context_len": int(m),
                                "target_index": int(t),
                                "episode": int(episode_offset + i),
                                "p_true": float(p_true[i]),
                                "y_test": int(y_test[i]),
                                "parents_cfg": int(cfg[i]),
                                "p_hat_model": float(p_hat_model[i]),
                                "tv_model": float(tv_model[i]),
                                "p_hat_naive": float(p_hat_naive[i]),
                                "tv_naive": float(tv_naive[i]),
                                "p_hat_bayes": float(p_hat_bayes[i]),
                                "tv_bayes": float(tv_bayes[i]),
                            }
                        )

                episode_offset += B
                remaining -= B


def evaluate_tv_over_context(
    model: torch.nn.Module,
    template: BNTemplate,
    p1_list_fixed: List[np.ndarray],
    spec: EvalSpec,
) -> None:
    """Writes a CSV with per-episode model predictions and ground truth (no baselines)."""
    N = template.num_nodes
    if len(p1_list_fixed) != N:
        raise BNError("p1_list_fixed must have length equal to template.num_nodes")
    num_graphs = p1_list_fixed[0].shape[0]
    for i, parents in enumerate(template.parent_idx):
        k = int(parents.size)
        K = 1 << k
        if p1_list_fixed[i].shape != (num_graphs, K):
            raise BNError(f"p1_list_fixed[{i}] must have shape ({num_graphs},{K}), got {p1_list_fixed[i].shape}")

    rng = np.random.default_rng(spec.seed)

    device = torch.device(spec.device)
    model = model.to(device)
    model.eval()

    out_path = Path(spec.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

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

        for m in spec.context_lens:
            L = int(m) + 1
            remaining = int(spec.num_episodes)
            episode_offset = 0

            while remaining > 0:
                B = min(remaining, spec.infer_batch_size)

                graph_ids = rng.integers(0, num_graphs, size=B, dtype=np.int64)

                X_full = sample_many_graphs(
                    template=template,
                    p1_list=p1_list_fixed,
                    graph_ids=graph_ids,
                    num_examples=L,
                    rng=rng,
                )

                for t in range(N):
                    X_out = _build_icl_x(X_full, target_index=t)

                    y_test = X_full[:, L - 1, t].astype(np.int64)

                    test_prefix = X_full[:, L - 1, :]
                    parents_idx = template.parent_idx[t]
                    cfg = _compute_parent_cfg(test_prefix, parents_idx)
                    p_true = p1_list_fixed[t][graph_ids, cfg].astype(np.float64)

                    x_tensor = torch.as_tensor(X_out, dtype=torch.float32, device=device)
                    with torch.no_grad():
                        logits = model(x_tensor)
                        p_hat = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float64)

                    tv = np.abs(p_hat - p_true)

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
