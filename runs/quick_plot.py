import argparse
import math
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

parser = argparse.ArgumentParser(
    description="Plot TV vs num_examples with std across seeds",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--graph", type=str, default="tree", help="graph: tree, general, chain")
parser.add_argument("--num_examples", type=int, default=10)
args = parser.parse_args()

SEEDS = [1111]
# 2222]
# , 3333, 4444, 5555]

rows = []
for seed in SEEDS:
    file_path = Path(
        f"/home/dennis/JASACODE/runs/{args.graph}/seed_{seed}/{args.num_examples}/20000_eval_tv.csv"
    )
    df = pd.read_csv(file_path)

    # Aggregate per (context_len, target_index) for THIS seed
    seed_vals = (
        df.groupby(["context_len", "target_index"], as_index=False)[
            ["tv_model", "tv_naive", "tv_bayes"]
        ]
        .mean()
        .rename(columns={"context_len": "num_examples"})
    )
    seed_vals["seed"] = seed
    rows.append(seed_vals)

plot_df = pd.concat(rows, ignore_index=True)

# Convert to long format: one row per (num_examples, target, seed, method)
plot_df = plot_df.melt(
    id_vars=["num_examples", "target_index", "seed"],
    value_vars=["tv_model", "tv_naive", "tv_bayes"],
    var_name="method",
    value_name="tv",
)

# Optional: nicer labels
method_map = {
    "tv_model": "Transformer",
    "tv_naive": "Naive",
    "tv_bayes": "Bayes (known DAG)",
}
plot_df["method"] = plot_df["method"].map(method_map)


# ---------- sanity checks (optional but recommended) ----------
# Check how many seeds contributed to each (num_examples, target_index)
counts = (
    plot_df.groupby(["num_examples", "target_index"])["seed"]
           .nunique()
           .reset_index(name="n_seeds")
)
bad = counts[counts["n_seeds"] != len(SEEDS)]
if not bad.empty:
    print("WARNING: some (num_examples, target_index) groups do not have all seeds:")
    print(bad.sort_values(["target_index", "num_examples"]).to_string(index=False))

# If you want to see the actual 5 values for a given group:
# print(plot_df[(plot_df["target_index"]==0) & (plot_df["num_examples"]==10)].sort_values("seed"))
# -------------------------------------------------------------

target_indices = sorted(plot_df["target_index"].unique())
n_targets = len(target_indices)

ncols = 3
nrows = math.ceil(n_targets / ncols)

fig, axes = plt.subplots(
    nrows=nrows, ncols=ncols,
    figsize=(5.0 * ncols, 3.8 * nrows),
    sharex=True, sharey=True
)
axes = axes.flatten()

for ax, target_idx in zip(axes, target_indices):
    tdf = plot_df[plot_df["target_index"] == target_idx].sort_values("num_examples")

    sns.lineplot(
        data=tdf,
        x="num_examples",
        y="tv",
        hue="method",
        estimator="mean",   # mean across seeds
        errorbar="sd",      # std across seeds
        marker="o",
        ax=ax
    )

    ax.set_title(f"Target {target_idx}")
    ax.set_xlabel("Number of Examples")
    ax.set_ylabel("TV Distance")
    ax.grid(True, alpha=0.3)


handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles, labels,
    loc="upper center",
    ncol=3,
    frameon=False,
)

for ax in axes:
    leg = ax.get_legend()
    if leg is not None:
        leg.remove()

for ax in axes[len(target_indices):]:
    ax.axis("off")

fig.tight_layout()
out_path = f"/home/dennis/JASACODE/runs/{args.graph}/num_examples_{args.num_examples}_tv.png"
plt.savefig(out_path, dpi=640)
print(f"Saved: {out_path}")
