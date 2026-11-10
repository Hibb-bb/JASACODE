"""
Aggregate and plot the rebuttal experiments (runs_rebuttal/).

Per-config 2x2 figures (same layout as quick_plot.py) and, per graph and loss,
a summary overlay of node-averaged TV for several model configs vs the two baselines.

Usage:
  python plot_rebuttal.py                 # everything found under runs_rebuttal/
  python plot_rebuttal.py --out imgs/rebuttal
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SEEDS = [1111, 2222, 3333, 4444, 5555]
FONT_SIZE = 20
PLOT_RC = {
    "font.size": FONT_SIZE, "axes.titlesize": FONT_SIZE, "axes.labelsize": FONT_SIZE,
    "xtick.labelsize": FONT_SIZE - 4, "ytick.labelsize": FONT_SIZE - 4,
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
}
GRAPH_TITLE = {
    "tree": "Tree (7 Node)", "chain": "Chain (7 Node)", "general": "General (5 Node)",
    "tree5": "Tree (5 Node)", "chain5": "Chain (5 Node)", "general7": "General (7 Node)",
}
MODEL_LABEL = {
    "L4": "Transformer (L4, H8, paper)",
    "gpt2_L2H1D256": "Transformer (L2, H1)",
    "theory_L2H1D30": "Transformer (L2, H1, Thm, width 30)",
    "theory_L2H1D22": "Transformer (L2, H1, Thm, width 22)",
    "theory_L2H1D64": "Transformer (L2, H1, Thm, width 64)",
    "theory_L2H1D256": "Transformer (L2, H1, Thm, width 256)",
}
LOSS_TITLE = {"tv": "TV loss on empirical CPT", "bce": "Cross-entropy on realized value"}
LOSS_SHORT = {"tv": "Trained with TV", "bce": "Trained with CE"}
# theorem-width theory models, (2M+1)d: 30 for the 7-node graphs, 22 for the 5-node general graph
THEORY_PAPER_WIDTH = ["theory_L2H1D30", "theory_L2H1D22"]
# Labels without the width, used for the individual figures and the 3-curve summaries.
MODEL_LABEL_SHORT = {**MODEL_LABEL, **{t: "Transformer (L2, H1, Thm)" for t in THEORY_PAPER_WIDTH}}


def _flatten(c):
    return f"{c[0]}_{c[1]}" if isinstance(c, tuple) and c[1] else (c[0] if isinstance(c, tuple) else c)


def discover(root: Path) -> Dict[Tuple[str, str, str], Dict[int, Path]]:
    """-> {(loss, graph, model_tag): {seed: csv_path}}"""
    found: Dict[Tuple[str, str, str], Dict[int, Path]] = {}
    for csv in root.rglob("*_eval_tv.csv"):
        rel = csv.relative_to(root).parts
        loss, graph = rel[0], rel[1]
        m = re.match(r"seed_(\d+)", rel[2])
        if not m:
            continue
        seed = int(m.group(1))
        model_tag = rel[4]
        found.setdefault((loss, graph, model_tag), {})[seed] = csv
    return found


# Metric being plotted. Internally the columns keep the names tv_model / tv_naive / tv_bayes;
# with --metric kl they hold KL(p_true || p_hat) computed from the stored probabilities.
METRIC = {"name": "tv", "ylabel": "TV Distance", "log_y": False}


def _apply_metric(df: pd.DataFrame) -> pd.DataFrame:
    if METRIC["name"] == "tv":
        return df
    from utils.evaluate import kl_divergence_safe
    df = df.copy()
    if "p_hat_model" in df.columns:  # binary CSVs store Bernoulli parameters
        for src, dst in (("p_hat_model", "tv_model"), ("p_hat_naive", "tv_naive"), ("p_hat_bayes", "tv_bayes")):
            df[dst] = kl_divergence_safe(df["p_true"].values, df[src].values)
    else:  # categorical CSVs carry kl_* columns
        for k in ("model", "naive", "bayes"):
            df[f"tv_{k}"] = df[f"kl_{k}"]
    return df


def _finish_axis(ax) -> None:
    if METRIC["log_y"]:
        ax.set_yscale("log")


def load_group(paths: Dict[int, Path]) -> pd.DataFrame:
    rows = []
    for seed, p in sorted(paths.items()):
        df = _apply_metric(pd.read_csv(p))
        v = (df.groupby(["context_len", "target_index"], as_index=False)[["tv_model", "tv_naive", "tv_bayes"]]
             .mean().rename(columns={"context_len": "num_examples"}))
        v["seed"] = seed
        rows.append(v)
    return pd.concat(rows, ignore_index=True)


def plot_2x2(wide: pd.DataFrame, title: str, out: Path) -> None:
    agg = wide.groupby(["num_examples", "target_index"])[["tv_model", "tv_naive", "tv_bayes"]].agg(["mean", "std"]).reset_index()
    agg.columns = [_flatten(c) for c in agg.columns]
    per_seed = wide.groupby(["num_examples", "seed"], as_index=False)[["tv_model", "tv_naive", "tv_bayes"]].mean()
    avg = per_seed.groupby("num_examples")[["tv_model", "tv_naive", "tv_bayes"]].agg(["mean", "std"]).reset_index()
    avg.columns = [_flatten(c) for c in avg.columns]

    plt.rcParams.update(PLOT_RC)
    targets = sorted(agg["target_index"].unique())
    colors = plt.cm.tab10(np.arange(len(targets)))
    fig, axes = plt.subplots(2, 2, figsize=(10, 9), sharex=True, sharey=True)

    def panel(ax, col, ttl, legend):
        for i, t in enumerate(targets):
            sub = agg[agg["target_index"] == t].sort_values("num_examples")
            yerr = sub[f"{col}_std"].values
            yerr = None if np.any(np.isnan(yerr)) else yerr
            ax.errorbar(sub["num_examples"], sub[f"{col}_mean"], yerr=yerr, marker="o",
                        label=f"Node {t}" if legend else None, color=colors[i], alpha=0.7, capsize=3)
        ax.set_title(ttl); ax.grid(True, alpha=0.3)
        if legend:
            ax.legend(fontsize=14, loc="upper right")

    panel(axes[0, 0], "tv_model", "Transformer", True); axes[0, 0].set_ylabel(METRIC["ylabel"])
    panel(axes[0, 1], "tv_bayes", "Bayesian Inference", False)
    panel(axes[1, 0], "tv_naive", "Naive Inference", False)
    axes[1, 0].set_ylabel(METRIC["ylabel"]); axes[1, 0].set_xlabel("Number of Examples")
    ax = axes[1, 1]
    for col, lab in [("tv_model", "Transformer"), ("tv_naive", "Naive"), ("tv_bayes", "Bayes")]:
        sub = avg.sort_values("num_examples")
        yerr = sub[f"{col}_std"].values
        yerr = None if np.any(np.isnan(yerr)) else yerr
        ax.errorbar(sub["num_examples"], sub[f"{col}_mean"], yerr=yerr, marker="o", label=lab, capsize=3)
    for a in axes.flat:
        _finish_axis(a)
    ax.set_title("Averaged Across Nodes"); ax.set_xlabel("Number of Examples"); ax.grid(True, alpha=0.3)
    ax.legend(fontsize=14, loc="upper right")
    for a, lab in zip(axes.flat, ["(a)", "(b)", "(c)", "(d)"]):
        a.text(0.02, 0.98, lab, transform=a.transAxes, fontsize=FONT_SIZE, fontweight="bold", va="top", ha="left")
    fig.suptitle(title, fontsize=FONT_SIZE + 2, y=0.93)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300, bbox_inches="tight"); plt.close()
    print("saved", out)


def plot_summary(groups: Dict[str, pd.DataFrame], title: str, out: Path,
                 naive_label: str = "Naive (graph-agnostic)",
                 labels: Dict[str, str] | None = None) -> None:
    """Overlay node-averaged tv_model for several model tags; baselines from the first group."""
    labels = {**MODEL_LABEL, **(labels or {})}
    plt.rcParams.update(PLOT_RC)
    fig, ax = plt.subplots(figsize=(8, 6))
    first = True
    for tag, wide in groups.items():
        per_seed = wide.groupby(["num_examples", "seed"], as_index=False)[["tv_model", "tv_naive", "tv_bayes"]].mean()
        avg = per_seed.groupby("num_examples")[["tv_model", "tv_naive", "tv_bayes"]].agg(["mean", "std"]).reset_index()
        avg.columns = [_flatten(c) for c in avg.columns]
        avg = avg.sort_values("num_examples")
        if first:
            ax.plot(avg["num_examples"], avg["tv_naive_mean"], "k--", marker="x", label=naive_label)
            ax.plot(avg["num_examples"], avg["tv_bayes_mean"], "k-", marker="s", label="Bayesian inference")
            first = False
        yerr = avg["tv_model_std"].values
        yerr = None if np.any(np.isnan(yerr)) else yerr
        ax.errorbar(avg["num_examples"], avg["tv_model_mean"], yerr=yerr, marker="o", capsize=3,
                    label=labels.get(tag, tag))
    ax.set_xlabel("Number of Examples"); ax.set_ylabel(f"{METRIC['ylabel']} (avg. over nodes)"); _finish_axis(ax)
    ax.set_title(title, fontsize=FONT_SIZE); ax.grid(True, alpha=0.3); ax.legend(fontsize=12)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); plt.savefig(out, dpi=300, bbox_inches="tight"); plt.close()
    print("saved", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs_rebuttal")
    ap.add_argument("--out", default="imgs/rebuttal")
    ap.add_argument("--summary-csv", default=None, help="default: <out>/summary.csv")
    ap.add_argument("--metric", choices=["tv", "kl"], default="tv",
                    help="kl = KL(p_true || p_hat) in nats, predictions clipped to [1e-6, 1-1e-6]")
    ap.add_argument("--log-y", action="store_true", help="log-scale y axis")
    args = ap.parse_args()
    METRIC["log_y"] = args.log_y
    if args.metric == "kl":
        METRIC.update(name="kl", ylabel="KL Divergence")
    if args.summary_csv is None:
        args.summary_csv = str(Path(args.out) / "summary.csv")
    root, out = Path(args.root), Path(args.out)
    found = discover(root)
    if not found:
        raise SystemExit(f"no eval CSVs under {root}")

    loaded: Dict[Tuple[str, str, str], pd.DataFrame] = {}
    summary_rows = []
    for key, paths in sorted(found.items()):
        loss, graph, tag = key
        missing = [s for s in SEEDS if s not in paths]
        if missing:
            print(f"[{loss}/{graph}/{tag}] missing seeds {missing} (using {len(paths)})")
        wide = load_group(paths)
        loaded[key] = wide
        plot_2x2(wide, f"{GRAPH_TITLE.get(graph, graph)} - {MODEL_LABEL_SHORT.get(tag, tag)}\n{LOSS_TITLE[loss]}",
                 out / loss / f"{graph}_{tag}.png")
        per_seed = wide.groupby(["num_examples", "seed"], as_index=False)[["tv_model", "tv_naive", "tv_bayes"]].mean()
        for n, sub in per_seed.groupby("num_examples"):
            summary_rows.append(dict(loss=loss, graph=graph, model=tag, num_examples=n, n_seeds=len(paths),
                                     tv_model=sub.tv_model.mean(), tv_model_std=sub.tv_model.std(),
                                     tv_naive=sub.tv_naive.mean(), tv_bayes=sub.tv_bayes.mean()))
    pd.DataFrame(summary_rows).to_csv(args.summary_csv, index=False)
    print("summary ->", args.summary_csv)

    for loss in sorted({k[0] for k in loaded}):
        for graph in sorted({k[1] for k in loaded if k[0] == loss}):
            groups = {k[2]: v for k, v in loaded.items() if k[0] == loss and k[1] == graph}
            order = ["L4", "gpt2_L2H1D256"] + sorted(t for t in groups if t.startswith("theory"))
            groups = {t: groups[t] for t in order if t in groups}
            plot_summary(groups, f"{GRAPH_TITLE.get(graph, graph)}: {LOSS_TITLE[loss]}",
                         out / f"summary_{loss}_{graph}.png")
            # Minimal figure: Naive, Bayesian inference, and the theorem model at the theorem width.
            thm = [t for t in THEORY_PAPER_WIDTH if t in groups]
            if thm:
                plot_summary({thm[0]: groups[thm[0]]},
                             f"{GRAPH_TITLE.get(graph, graph)}: {LOSS_SHORT[loss]}",
                             out / f"summary3_{loss}_{graph}.png", naive_label="Naive", labels=MODEL_LABEL_SHORT)


if __name__ == "__main__":
    main()
