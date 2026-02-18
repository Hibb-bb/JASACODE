#!/usr/bin/env python3
"""
Plot TV distance vs num_examples for random-DAG training.

Aggregates evaluation results across seeds (mean +/- std) and produces
a 2x2 figure:
  (a) Transformer — per-node lines
  (b) Averaged across nodes — all 3 methods
  (c) Naive inference — per-node lines
  (d) Bayesian — per-node lines

Supports two evaluation modes:
  --eval-type random_dags   (default) — aggregate over random DAG eval CSVs
  --eval-type fixed         — aggregate over fixed structure eval CSVs (tree/chain/general)
"""
import argparse
import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


parser = argparse.ArgumentParser(
    description="Plot TV vs num_examples (2x2, with std across seeds) for random-DAG training",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--output-dir", type=str, required=True,
                    help="Base output directory (e.g. runs/best/random_dag_5nodes_p0.1to0.8_ctx500)")
parser.add_argument("--run-prefix", type=str, default="rdag",
                    help="Run name prefix (before _seed)")
parser.add_argument("--run-suffix", type=str, required=True,
                    help="Run name suffix after seed (e.g. 5nodes_p=0.1to0.8_ctx500)")
parser.add_argument("--eval-type", type=str, default="random_dags",
                    choices=["random_dags", "fixed"],
                    help="Which eval results to plot")
parser.add_argument("--structure", type=str, default="tree",
                    help="Structure name for --eval-type fixed (tree, chain, general)")
parser.add_argument("--num-nodes", type=int, default=5,
                    help="Number of nodes (used in the plot title)")
args = parser.parse_args()


# ── Style ───────────────────────────────────────────────────────────────
font_size = 20

plt.rcParams.update({
    'font.size': font_size,
    'axes.titlesize': font_size,
    'axes.labelsize': font_size,
    'xtick.labelsize': font_size - 4,
    'ytick.labelsize': font_size - 4,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
})


# ── Load data across seeds ──────────────────────────────────────────────
SEEDS = [1111, 2222, 3333, 4444, 5555]

rows = []
for seed in SEEDS:
    # Build eval directory path:
    #   {output_dir}/{run_prefix}_seed{seed}_{run_suffix}_eval/{eval_subdir}/
    eval_base = Path(args.output_dir) / f"{args.run_prefix}_seed{seed}_{args.run_suffix}_eval"

    if args.eval_type == "random_dags":
        eval_dir = eval_base / "random_dags"
        if not eval_dir.exists():
            print(f"WARNING: {eval_dir} not found, skipping seed {seed}")
            continue
        # Load all eval_tv_dag*.csv files and concatenate
        csv_files = sorted(eval_dir.glob("eval_tv_dag*.csv"))
        if not csv_files:
            print(f"WARNING: no CSV files in {eval_dir}, skipping seed {seed}")
            continue
        dfs = []
        for f in csv_files:
            dfs.append(pd.read_csv(f))
        df = pd.concat(dfs, ignore_index=True)
    else:
        # Fixed structure eval
        eval_dir = eval_base / "fixed_structures"
        csv_path = eval_dir / f"eval_tv_{args.structure}.csv"
        if not csv_path.exists():
            print(f"WARNING: {csv_path} not found, skipping seed {seed}")
            continue
        df = pd.read_csv(csv_path)

    # Aggregate per (context_len, target_index) for this seed (mean over episodes/DAGs)
    seed_vals = (
        df.groupby(["context_len", "target_index"], as_index=False)[
            ["tv_model", "tv_naive", "tv_bayes"]
        ]
        .mean()
        .rename(columns={"context_len": "num_examples"})
    )
    seed_vals["seed"] = seed
    rows.append(seed_vals)

if not rows:
    raise FileNotFoundError(
        f"No eval data found in {args.output_dir} for eval_type={args.eval_type}"
    )

wide_df = pd.concat(rows, ignore_index=True)
print(f"Loaded {len(rows)} seed(s) for eval_type='{args.eval_type}'")


# ── Aggregate across seeds ──────────────────────────────────────────────
agg = (
    wide_df.groupby(["num_examples", "target_index"])[["tv_model", "tv_naive", "tv_bayes"]]
    .agg(["mean", "std"])
    .reset_index()
)

def _flatten_col(c):
    if isinstance(c, tuple):
        return f"{c[0]}_{c[1]}" if c[1] else c[0]
    return c

agg.columns = [_flatten_col(c) for c in agg.columns]


# ── For top-right panel: average across target_index first (per seed) ───
avg_per_seed = (
    wide_df.groupby(["num_examples", "seed"], as_index=False)[["tv_model", "tv_naive", "tv_bayes"]]
    .mean()
)
avg_agg = (
    avg_per_seed.groupby("num_examples")[["tv_model", "tv_naive", "tv_bayes"]]
    .agg(["mean", "std"])
    .reset_index()
)
avg_agg.columns = [_flatten_col(c) for c in avg_agg.columns]


# ── Plot 2x2 figure ─────────────────────────────────────────────────────
target_indices = sorted(agg["target_index"].unique())
n_targets = len(target_indices)
target_colors = plt.cm.tab10(np.arange(n_targets))

fig, axes = plt.subplots(
    nrows=2, ncols=2,
    figsize=(10, 9),
    sharex=True, sharey=True,
)


def plot_panel(ax, agg_df, value_mean_col, value_std_col, title, show_legend=True):
    """Plot one panel: one line per target_index with error bars (std)."""
    for i, target_idx in enumerate(target_indices):
        sub = agg_df[agg_df["target_index"] == target_idx].sort_values("num_examples")
        x = sub["num_examples"].values
        y = sub[value_mean_col].values
        yerr = sub[value_std_col].values
        if np.any(np.isnan(yerr)):
            yerr = None
        ax.errorbar(
            x, y, yerr=yerr,
            marker="o", label=f"Node {target_idx}" if show_legend else None,
            color=target_colors[i], alpha=0.7,
            capsize=3, capthick=1,
        )
    ax.set_title(title)
    ax.set_xlabel("Number of Examples")
    ax.set_ylabel("TV Distance")
    ax.grid(True, alpha=0.3)
    if show_legend:
        ax.legend(fontsize=14, loc="upper right")


# (a) Transformer per-node
plot_panel(axes[0, 0], agg, "tv_model_mean", "tv_model_std", "Transformer", show_legend=True)
axes[0, 0].set_xlabel("")

# (b) Averaged across nodes — all 3 methods
ax_tr = axes[0, 1]
for col, label in [
    ("tv_model", "Transformer"),
    ("tv_naive", "Naive"),
    ("tv_bayes", "Bayes"),
]:
    sub = avg_agg.sort_values("num_examples")
    x = sub["num_examples"].values
    y = sub[f"{col}_mean"].values
    yerr = sub[f"{col}_std"].values
    if np.any(np.isnan(yerr)):
        yerr = None
    ax_tr.errorbar(x, y, yerr=yerr, marker="o", label=label, capsize=3, capthick=1)
ax_tr.set_title("Averaged Across Nodes")
ax_tr.set_xlabel("")
ax_tr.set_ylabel("")
ax_tr.grid(True, alpha=0.3)
ax_tr.legend(fontsize=14, loc="upper right")

# (c) Naive per-node
plot_panel(axes[1, 0], agg, "tv_naive_mean", "tv_naive_std", "Naive Inference", show_legend=False)

# (d) Bayesian per-node
plot_panel(axes[1, 1], agg, "tv_bayes_mean", "tv_bayes_std", "Bayesian", show_legend=False)
axes[1, 1].set_ylabel("")

# Panel labels
for ax, label in zip(axes.flat, ["(a)", "(b)", "(c)", "(d)"]):
    ax.text(0.02, 0.98, label, transform=ax.transAxes,
            fontsize=font_size, fontweight="bold", va="top", ha="left")

# Overall title
if args.eval_type == "random_dags":
    title_str = f"Random DAG ({args.num_nodes} Nodes)"
else:
    structure_label = args.structure.replace("_", " ").title()
    title_str = f"Random DAG — {structure_label} Generalization ({args.num_nodes} Nodes)"

fig.suptitle(title_str, fontsize=font_size + 6, y=0.90)
fig.tight_layout(rect=[0, 0, 1, 0.96])

# Output path
if args.eval_type == "random_dags":
    out_path = Path(args.output_dir) / "final_eval_random_dags.png"
else:
    out_path = Path(args.output_dir) / f"final_eval_fixed_{args.structure}.png"

out_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out_path, dpi=600, bbox_inches="tight")
plt.close()
print(f"Saved: {out_path}")
