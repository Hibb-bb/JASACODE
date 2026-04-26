from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Dict, Any, Optional

import numpy as np
import torch

from data.bn_template import BNTemplate
from data.categorical_template import CategoricalTemplate
from data.multigraph_sampler import sample_many_graphs, sample_many_graphs_categorical
from data.binary_bn import BNError


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


def _compute_prefix_cfg_binary(rows: np.ndarray) -> np.ndarray:
    """Config id for prefix (x_0..x_{t-1}); rows (B, nprev), binary digits."""
    nprev = int(rows.shape[1])
    # Masked values use -1 sentinel; treat as 0 for config id computation.
    if rows.size:
        rows = np.where(rows < 0, 0, rows)
    weights = (1 << np.arange(nprev, dtype=np.int64))[None, :]
    return (rows.astype(np.int64) * weights).sum(axis=1)


def _compute_prefix_cfg_categorical_rows(rows: np.ndarray, K: int) -> np.ndarray:
    """Config id = sum_j x_j * K^j; rows (B, nprev). nprev==0 -> zeros(B)."""
    if rows.shape[1] == 0:
        return np.zeros((rows.shape[0],), dtype=np.int64)
    # Masked values use -1 sentinel; treat as 0 for config id computation.
    if rows.size:
        rows = np.where(rows < 0, 0, rows)
    powers = np.power(K, np.arange(rows.shape[1], dtype=np.int64))
    return (rows.astype(np.int64) * powers[None, :]).sum(axis=1)


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
    # Masked values use -1 sentinel; treat as 0 for config id computation.
    if pv.size:
        pv = np.where(pv < 0, 0, pv)
    weights = (1 << np.arange(k, dtype=np.int64))[None, :]  # (1,k)
    return (pv * weights).sum(axis=1)


def _icl_masked_x(X_full: np.ndarray, target_index: int) -> np.ndarray:
    """Same node masking as the model ICL input (without the appended target_index channel).

    Context rows: zero columns t+1..N-1. Test row: zero columns t..N-1.
    """
    B, L, N = X_full.shape
    t = int(target_index)
    # X_full is often uint8; -1 mask sentinel requires signed dtype.
    X_mask = X_full.astype(np.int64, copy=True)
    if t + 1 < N:
        X_mask[:, : L - 1, t + 1 :] = -1
    X_mask[:, L - 1, t:] = -1
    return X_mask


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
    X_mask = _icl_masked_x(X_full, target_index)
    tgt_feat = np.full((B, L, 1), t, dtype=np.int64)
    X_out = np.concatenate([X_mask, tgt_feat], axis=2)
    return X_out  # (B, L, N+1)


def _tv_abs_binary_safe(p_hat: np.ndarray, p_true: np.ndarray) -> np.ndarray:
    """|p_hat - p_true|; non-finite -> 1.0 (max TV for a Bernoulli parameter)."""
    tv = np.abs(p_hat - p_true)
    return np.where(np.isfinite(tv), tv, 1.0)


def _tv_categorical_safe(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """0.5 * sum |p - q|; non-finite -> 1.0."""
    tv = 0.5 * np.abs(p - q).sum(axis=1)
    return np.where(np.isfinite(tv), tv, 1.0)


def _naive_graph_agnostic_from_context_bn(
    X_full: np.ndarray,
    target_index: int,
) -> np.ndarray:
    """
    Graph-agnostic naive: MLE P(X_t=1 | x_0..x_{t-1}) from masked context (ignores true DAG).
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


def _naive_context_tot_at_prefix_bn(X_full: np.ndarray, target_index: int) -> np.ndarray:
    """Context rows whose masked prefix matches masked test prefix (graph-agnostic naive). (B,) int64."""
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
    X_full: np.ndarray,        # (B, L, N)
    template: BNTemplate,
    target_index: int,         # t
) -> np.ndarray:
    """
    Bayesian CPT baseline from ICL context with the same masking as the model.
    MLE: P(X_t=1|cfg) = count_1(cfg)/count(cfg); missing cfg or zero count -> nan.
    cfg on context rows and test row from masked tensors.
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


def _bayes_context_tot_at_test_cfg_bn(
    X_full: np.ndarray,
    template: BNTemplate,
    target_index: int,
) -> np.ndarray:
    """Count of masked-context rows whose parent cfg equals masked-test cfg. (B,) int64."""
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


def _bayes_context_tot_at_test_cfg_cat(
    X_full: np.ndarray,
    template: CategoricalTemplate,
    target_index: int,
) -> np.ndarray:
    """Same as _bayes_context_tot_at_test_cfg_bn for categorical parent configs (masked)."""
    B, L, N = X_full.shape
    m = L - 1
    t = int(target_index)
    K = template.cardinality
    parents_idx = template.parent_idx[t]
    k = int(parents_idx.size)
    num_configs = K ** k

    X_mask = _icl_masked_x(X_full, target_index)
    X_ctx = X_mask[:, :m, :]
    X_ctx_flat = X_ctx.reshape(B * m, N)
    cfg_ctx = _compute_parent_cfg_categorical(
        X_ctx_flat, parents_idx, K
    ).reshape(B, m)
    cfg_test = _compute_parent_cfg_categorical(X_mask[:, L - 1, :], parents_idx, K)

    out = np.empty((B,), dtype=np.int64)
    for i in range(B):
        tot = np.bincount(cfg_ctx[i], minlength=num_configs).astype(np.int64)
        out[i] = tot[int(cfg_test[i])]
    return out


def _compute_parent_cfg_categorical(
    test_prefix: np.ndarray,
    parent_idx: np.ndarray,
    K: int,
) -> np.ndarray:
    """Parent config id = sum parent_val[j] * K^j. Returns (B,) int64."""
    k = int(parent_idx.size)
    if k == 0:
        return np.zeros((test_prefix.shape[0],), dtype=np.int64)
    pv = test_prefix[:, parent_idx].astype(np.int64)
    powers = np.power(K, np.arange(k, dtype=np.int64))
    return (pv * powers[None, :]).sum(axis=1)


def _naive_graph_agnostic_from_context_categorical(
    X_full: np.ndarray,
    template: CategoricalTemplate,
    target_index: int,
) -> np.ndarray:
    """
    Graph-agnostic naive: MLE P(X_t | x_0..x_{t-1}) from masked context (full prefix, not DAG).
    t==0 -> marginal histogram / m. Zero matching count -> nan row. (B, K).
    """
    B, L, N = X_full.shape
    m = L - 1
    t = int(target_index)
    K = template.cardinality
    X_mask = _icl_masked_x(X_full, target_index)
    X_ctx = X_mask[:, :m, :]
    y_ctx = X_ctx[:, :, t].astype(np.int64)
    nprev = t
    num_configs = int(K**nprev) if nprev > 0 else 1

    if nprev == 0:
        cfg_ctx = np.zeros((B, m), dtype=np.int64)
        cfg_test = np.zeros((B,), dtype=np.int64)
    else:
        flat = X_ctx[:, :, :nprev].reshape(B * m, nprev)
        cfg_ctx = _compute_prefix_cfg_categorical_rows(flat, K).reshape(B, m)
        cfg_test = _compute_prefix_cfg_categorical_rows(X_mask[:, L - 1, :nprev], K)

    out = np.empty((B, K), dtype=np.float64)
    for i in range(B):
        tot = np.bincount(cfg_ctx[i], minlength=num_configs).astype(np.float64)
        counts = np.zeros((num_configs, K), dtype=np.float64)
        for v in range(K):
            counts[:, v] = np.bincount(
                cfg_ctx[i],
                weights=(y_ctx[i] == v).astype(np.float64),
                minlength=num_configs,
            )
        c = int(cfg_test[i])
        if tot[c] > 0:
            out[i] = counts[c, :] / tot[c]
        else:
            out[i] = np.nan
    return out


def _naive_context_tot_at_prefix_cat(
    X_full: np.ndarray,
    template: CategoricalTemplate,
    target_index: int,
) -> np.ndarray:
    """Context rows matching masked test prefix cfg (graph-agnostic naive). (B,) int64."""
    B, L, N = X_full.shape
    m = L - 1
    t = int(target_index)
    K = template.cardinality
    nprev = t
    num_configs = int(K**nprev) if nprev > 0 else 1
    X_mask = _icl_masked_x(X_full, target_index)
    X_ctx = X_mask[:, :m, :]
    if nprev == 0:
        cfg_ctx = np.zeros((B, m), dtype=np.int64)
        cfg_test = np.zeros((B,), dtype=np.int64)
    else:
        flat = X_ctx[:, :, :nprev].reshape(B * m, nprev)
        cfg_ctx = _compute_prefix_cfg_categorical_rows(flat, K).reshape(B, m)
        cfg_test = _compute_prefix_cfg_categorical_rows(X_mask[:, L - 1, :nprev], K)

    out = np.empty((B,), dtype=np.int64)
    for i in range(B):
        tot = np.bincount(cfg_ctx[i], minlength=num_configs).astype(np.int64)
        out[i] = tot[int(cfg_test[i])]
    return out


def _bayes_cpt_from_context_categorical(
    X_full: np.ndarray,
    template: CategoricalTemplate,
    target_index: int,
) -> np.ndarray:
    """Bayesian CPT from masked context; MLE at masked-test cfg. Zero count -> nan row. (B, K)."""
    B, L, N = X_full.shape
    m = L - 1
    t = int(target_index)
    K = template.cardinality
    parents_idx = template.parent_idx[t]
    k = int(parents_idx.size)
    num_configs = K ** k

    X_mask = _icl_masked_x(X_full, target_index)
    X_ctx = X_mask[:, :m, :]
    y_ctx = X_ctx[:, :, t].astype(np.int64)
    X_ctx_flat = X_ctx.reshape(B * m, N)
    cfg_ctx = _compute_parent_cfg_categorical(
        X_ctx_flat, parents_idx, K
    ).reshape(B, m)
    cfg_test = _compute_parent_cfg_categorical(X_mask[:, L - 1, :], parents_idx, K)

    out = np.empty((B, K), dtype=np.float64)
    for i in range(B):
        tot = np.bincount(cfg_ctx[i], minlength=num_configs).astype(np.float64)
        counts = np.zeros((num_configs, K), dtype=np.float64)
        for v in range(K):
            counts[:, v] = np.bincount(
                cfg_ctx[i],
                weights=(y_ctx[i] == v).astype(np.float64),
                minlength=num_configs,
            )
        cfg = int(cfg_test[i])
        if tot[cfg] > 0:
            out[i] = counts[cfg, :] / tot[cfg]
        else:
            out[i] = np.nan
    return out


def _tv_categorical(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """TV between distributions: 0.5 * sum_i |p_i - q_i|. p, q (B, K); returns (B,)."""
    return 0.5 * np.abs(p - q).sum(axis=1)


def evaluate_tv_over_context_categorical_with_baselines(
    model: torch.nn.Module,
    template: CategoricalTemplate,
    cpt_list_fixed: List[np.ndarray],
    spec: EvalSpec,
) -> None:
    """
    Evaluate 3-class Sachs (or any categorical BN). TV = 0.5 * sum_c |p_c - q_c|.
    Writes CSV with tv_model, tv_naive, tv_bayes per episode.

    Naive: graph-agnostic MLE P(X_t | x_0..x_{t-1}) from masked context (ignores DAG).
    Bayes: MLE using true parents only. No smoothing; non-finite TV -> 1.0.
    If m=0, tv_naive is 1.0. If no context row matches the relevant cfg, tv_naive/tv_bayes -> 1.0.
    """
    N = template.num_nodes
    K = template.cardinality
    if len(cpt_list_fixed) != N:
        raise BNError("cpt_list_fixed length must match template.num_nodes")
    num_graphs = cpt_list_fixed[0].shape[0]
    for i, parents in enumerate(template.parent_idx):
        k = int(parents.size)
        ncfg = K ** k
        if cpt_list_fixed[i].shape != (num_graphs, ncfg, K):
            raise BNError(
                f"cpt_list_fixed[{i}] shape expected ({num_graphs},{ncfg},{K}), got {cpt_list_fixed[i].shape}"
            )

    rng = np.random.default_rng(spec.seed)
    device = torch.device(spec.device)
    model = model.to(device)
    model.eval()

    out_path = Path(spec.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "context_len", "target_index", "episode",
        "y_test", "parents_cfg",
        "tv_model", "tv_naive", "tv_bayes",
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
                X_full = sample_many_graphs_categorical(
                    template=template,
                    cpt_list=cpt_list_fixed,
                    graph_ids=graph_ids,
                    num_examples=L,
                    rng=rng,
                )

                for t in range(N):
                    y_test = X_full[:, L - 1, t].astype(np.int64)
                    test_prefix = X_full[:, L - 1, :]
                    parents_idx = template.parent_idx[t]
                    cfg = _compute_parent_cfg_categorical(test_prefix, parents_idx, K)
                    p_true = cpt_list_fixed[t][graph_ids, cfg, :].astype(np.float64)

                    X_out = _build_icl_x(X_full, target_index=t)
                    x_tensor = torch.as_tensor(X_out, dtype=torch.float32, device=device)
                    with torch.no_grad():
                        logits = model(x_tensor)
                        p_hat_model = torch.softmax(logits, dim=1).detach().cpu().numpy().astype(np.float64)

                    p_hat_naive = _naive_graph_agnostic_from_context_categorical(
                        X_full, template, t,
                    )
                    p_hat_bayes = _bayes_cpt_from_context_categorical(
                        X_full, template, t,
                    )

                    tv_model = _tv_categorical_safe(p_hat_model, p_true)
                    tv_naive = _tv_categorical_safe(p_hat_naive, p_true)
                    tv_bayes = _tv_categorical_safe(p_hat_bayes, p_true)

                    if m == 0:
                        tv_naive = np.ones_like(tv_naive)
                    naive_tot = _naive_context_tot_at_prefix_cat(X_full, template, t)
                    tv_naive = np.where(naive_tot == 0, 1.0, tv_naive)
                    bayes_tot = _bayes_context_tot_at_test_cfg_cat(
                        X_full, template, t,
                    )
                    tv_bayes = np.where(bayes_tot == 0, 1.0, tv_bayes)

                    for bi in range(B):
                        writer.writerow({
                            "context_len": int(m),
                            "target_index": int(t),
                            "episode": int(episode_offset + bi),
                            "y_test": int(y_test[bi]),
                            "parents_cfg": int(cfg[bi]),
                            "tv_model": float(tv_model[bi]),
                            "tv_naive": float(tv_naive[bi]),
                            "tv_bayes": float(tv_bayes[bi]),
                        })

                episode_offset += B
                remaining -= B


def evaluate_tv_over_context_with_baselines(
    model: torch.nn.Module,
    template: BNTemplate,
    p1_list_fixed: List[np.ndarray],
    spec: EvalSpec,
) -> None:
    """
    Naive: graph-agnostic P(X_t|x_0..x_{t-1}) from masked context. Bayes: true parents only.
    No smoothing; non-finite TV is 1.0. If m=0, tv_naive is 1.0. Zero cfg match -> TV 1.0.
    """
    # Basic checks - support multiple graphs for testing
    N = template.num_nodes
    if len(p1_list_fixed) != N:
        raise BNError("p1_list_fixed must have length equal to template.num_nodes")
    # Get number of graphs from first node's shape
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
        # ground truth
        "p_true",
        "y_test",
        "parents_cfg",
        # model
        "p_hat_model",
        "tv_model",
        # baselines
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

                # Randomly sample which graph to use for each batch element
                graph_ids = rng.integers(0, num_graphs, size=B, dtype=np.int64)
                X_full = sample_many_graphs(
                    template=template,
                    p1_list=p1_list_fixed,
                    graph_ids=graph_ids,
                    num_examples=L,
                    rng=rng,
                )  # (B, L, N)

                # For each target node, run everything
                for t in range(N):
                    # True label y_test
                    y_test = X_full[:, L - 1, t].astype(np.int64)  # (B,)

                    # Ground-truth conditional p_true from BN CPT (each batch element may use different graph)
                    test_prefix = X_full[:, L - 1, :]  # full values (parents live in <t)
                    parents_idx = template.parent_idx[t]
                    cfg = _compute_parent_cfg(test_prefix, parents_idx)  # (B,)
                    # Compute p_true using vectorized indexing: p1_list_fixed[t][graph_ids, cfg]
                    p_true = p1_list_fixed[t][graph_ids, cfg].astype(np.float64)  # (B,)

                    # ===== Model prediction =====
                    X_out = _build_icl_x(X_full, target_index=t)  # (B, L, N+1)
                    x_tensor = torch.as_tensor(X_out, dtype=torch.float32, device=device)
                    with torch.no_grad():
                        logits = model(x_tensor)  # expected (B,)
                        p_hat_model = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float64)
                    tv_model = _tv_abs_binary_safe(p_hat_model, p_true)

                    # ===== Baseline (1): graph-agnostic naive (prefix x_0..x_{t-1}) =====
                    p_hat_naive = _naive_graph_agnostic_from_context_bn(X_full, t).astype(np.float64)
                    tv_naive = _tv_abs_binary_safe(p_hat_naive, p_true)

                    # ===== Baseline (2): Bayes CPT (true parents only) =====
                    p_hat_bayes = _bayes_cpt_from_context(X_full, template, t).astype(np.float64)
                    tv_bayes = _tv_abs_binary_safe(p_hat_bayes, p_true)

                    if m == 0:
                        tv_naive = np.ones_like(tv_naive)
                    naive_tot = _naive_context_tot_at_prefix_bn(X_full, t)
                    tv_naive = np.where(naive_tot == 0, 1.0, tv_naive)
                    bayes_tot = _bayes_context_tot_at_test_cfg_bn(X_full, template, t)
                    tv_bayes = np.where(bayes_tot == 0, 1.0, tv_bayes)

                    # Write per-episode rows
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
    """
    Writes a CSV with per-episode predictions and ground truth.

    p1_list_fixed: length N; each entry shape (G, 2^k_i) for G different BNs.
    """
    # Basic checks - support multiple graphs for testing
    N = template.num_nodes
    if len(p1_list_fixed) != N:
        raise BNError("p1_list_fixed must have length equal to template.num_nodes")
    # Get number of graphs from first node's shape
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

                # Randomly sample which graph to use for each batch element
                graph_ids = rng.integers(0, num_graphs, size=B, dtype=np.int64)

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
                    # Compute p_true using vectorized indexing: p1_list_fixed[t][graph_ids, cfg]
                    p_true = p1_list_fixed[t][graph_ids, cfg].astype(np.float64)  # (B,)

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
