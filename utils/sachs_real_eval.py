from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from data.categorical_template import CategoricalTemplate

from .evaluate import (
    EvalSpec,
    _bayes_context_tot_at_test_cfg_cat,
    _build_icl_x,
    _compute_parent_cfg_categorical,
    _naive_graph_agnostic_from_context_categorical,
    _naive_context_tot_at_prefix_cat,
    _bayes_cpt_from_context_categorical,
    _tv_categorical_safe,
)


def _parse_interval_lower_bound(interval: str) -> float:
    """
    Parse the lower bound of an interval string like "[1.61,56.649]" or "(56.649,111.688]".
    Used only for ordering bins from low -> high so that codes 0,1,2 correspond
    to increasing levels.
    """
    s = interval.strip()
    if not s:
        return 0.0
    # Strip leading bracket/parenthesis
    if s[0] in "[(":
        s = s[1:]
    # Take substring before comma
    left = s.split(",")[0]
    try:
        return float(left)
    except ValueError:
        # Fallback: try to strip any remaining non-numeric characters
        filtered = "".join(ch for ch in left if (ch.isdigit() or ch in ".-+eE"))
        try:
            return float(filtered)
        except ValueError:
            return 0.0


def encode_disc_df_to_int(
    df: pd.DataFrame,
    expected_cardinality: int | None = 3,
) -> Tuple[np.ndarray, Dict[str, Dict[str, int]]]:
    """
    Encode discretized Sachs dataframe (interval strings) into integer codes 0..K-1 per column.

    Parameters
    ----------
    df:
        Dataframe whose columns are variables and whose entries are interval strings
        such as "[1.61,56.649]".
    expected_cardinality:
        If not None, emits a warning when the number of unique levels in a column
        does not match this value.

    Returns
    -------
    X:
        Array of shape (num_obs, N) with integer codes.
    mappings:
        Dict mapping column name -> {raw_label -> int_code}.
    """
    n = df.shape[0]
    cols = list(df.columns)
    X = np.zeros((n, len(cols)), dtype=np.int64)
    mappings: Dict[str, Dict[str, int]] = {}

    for j, col in enumerate(cols):
        vals = df[col].astype(str).to_numpy()
        uniq = np.unique(vals)
        # Order levels by numeric lower bound so 0,1,2 correspond to LOW->AVG->HIGH
        uniq_sorted = sorted(uniq, key=_parse_interval_lower_bound)
        if expected_cardinality is not None and len(uniq_sorted) != expected_cardinality:
            # Not fatal; just warn via print so the user can inspect.
            print(
                f"[encode_disc_df_to_int] Warning: column {col} has "
                f"{len(uniq_sorted)} levels (expected {expected_cardinality})."
            )
        mapping = {u: i for i, u in enumerate(uniq_sorted)}
        mappings[col] = mapping
        X[:, j] = np.vectorize(mapping.get)(vals)

    return X, mappings


def empirical_cpt_from_data(
    X_data: np.ndarray,           # (num_obs, N), int in {0..K-1}
    template: CategoricalTemplate,
) -> List[np.ndarray]:
    """
    Build empirical CPTs (MLE, no smoothing) for each node from the full dataset.

    CPT[cfg, v] = counts[cfg, v] / sum_v counts[cfg, v]; if the row sum is 0, use uniform 1/K.
    """
    N = template.num_nodes
    K = template.cardinality
    num_obs = X_data.shape[0]
    cpt_list: List[np.ndarray] = []
    uniform = np.full(K, 1.0 / K, dtype=np.float64)

    for t in range(N):
        parents_idx = template.parent_idx[t]
        k = int(parents_idx.size)
        if k == 0:
            counts = np.zeros((1, K), dtype=np.float64)
            y = X_data[:, t].astype(np.int64)
            for v in range(K):
                counts[0, v] = np.sum(y == v)
            row_sum = counts.sum()
            cpt = counts / row_sum if row_sum > 0 else uniform.reshape(1, K)
            cpt_list.append(cpt.astype(np.float64))
            continue

        num_configs = K ** k
        cfg_all = _compute_parent_cfg_categorical(
            X_data, parents_idx, K
        ).astype(np.int64)

        counts = np.zeros((num_configs, K), dtype=np.float64)
        y = X_data[:, t].astype(np.int64)

        for i in range(num_obs):
            cfg = int(cfg_all[i])
            v = int(y[i])
            counts[cfg, v] += 1.0

        row_sums = counts.sum(axis=1, keepdims=True)
        cpt = np.divide(counts, row_sums, out=np.zeros_like(counts), where=row_sums > 0)
        zero_rows = row_sums.squeeze() == 0
        if np.any(zero_rows):
            cpt[zero_rows, :] = uniform
        cpt_list.append(cpt.astype(np.float64))

    return cpt_list


def evaluate_tv_over_context_categorical_real(
    model: torch.nn.Module,
    template: CategoricalTemplate,
    X_data: np.ndarray,                 # (num_obs, N) real Sachs (encoded 0..K-1)
    cpt_emp_list: List[np.ndarray],     # empirical CPTs from empirical_cpt_from_data
    spec: EvalSpec,
    treatment_name: str,
) -> None:
    """
    Evaluate transformer on REAL Sachs data for a single treatment.

    Draws ICL episodes from X_data. Naive: P(X_t|x_0..x_{t-1}) graph-agnostic; Bayes: true parents.
    Same masking as the model; no smoothing; non-finite or zero-match TV -> 1.0.
    """
    N = template.num_nodes
    K = template.cardinality
    if len(cpt_emp_list) != N:
        raise ValueError("cpt_emp_list length must match template.num_nodes")

    num_obs = X_data.shape[0]
    if num_obs == 0:
        raise ValueError("X_data is empty; cannot evaluate on real Sachs data.")

    # Basic shape checks against template
    for i, parents in enumerate(template.parent_idx):
        k = int(parents.size)
        num_configs = K ** k
        expected_shape = (num_configs, K)
        if cpt_emp_list[i].shape != expected_shape:
            raise ValueError(
                f"cpt_emp_list[{i}] shape expected {expected_shape}, "
                f"got {cpt_emp_list[i].shape}"
            )

    device = torch.device(spec.device)
    model = model.to(device)
    model.eval()

    out_path = Path(spec.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "treatment",
        "context_len",
        "target_index",
        "episode",
        "y_test",
        "parents_cfg",
        "tv_model",
        "tv_naive",
        "tv_bayes",
    ]

    rng = np.random.default_rng(spec.seed)

    with out_path.open("w", newline="") as f:
        import csv

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for m in spec.context_lens:
            L = int(m) + 1
            remaining = int(spec.num_episodes)
            episode_offset = 0

            while remaining > 0:
                B = min(remaining, spec.infer_batch_size)
                # indices: (B, L)
                idx = rng.integers(0, num_obs, size=(B, L), dtype=np.int64)
                X_full = X_data[idx, :]  # (B, L, N)

                for t in range(N):
                    # Ground-truth CPT at test parent config, based on empirical CPT.
                    y_test = X_full[:, L - 1, t].astype(np.int64)
                    test_prefix = X_full[:, L - 1, :]
                    parents_idx = template.parent_idx[t]
                    cfg = _compute_parent_cfg_categorical(test_prefix, parents_idx, K)
                    p_true = cpt_emp_list[t][cfg, :].astype(np.float64)  # (B, K)

                    # Model prediction
                    X_out = _build_icl_x(X_full, target_index=t)
                    x_tensor = torch.as_tensor(
                        X_out, dtype=torch.float32, device=device
                    )
                    with torch.no_grad():
                        logits = model(x_tensor)
                        p_hat_model = (
                            torch.softmax(logits, dim=1)
                            .detach()
                            .cpu()
                            .numpy()
                            .astype(np.float64)
                        )

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
                        writer.writerow(
                            {
                                "treatment": treatment_name,
                                "context_len": int(m),
                                "target_index": int(t),
                                "episode": int(episode_offset + bi),
                                "y_test": int(y_test[bi]),
                                "parents_cfg": int(cfg[bi]),
                                "tv_model": float(tv_model[bi]),
                                "tv_naive": float(tv_naive[bi]),
                                "tv_bayes": float(tv_bayes[bi]),
                            }
                        )

                episode_offset += B
                remaining -= B

