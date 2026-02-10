import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

parser = argparse.ArgumentParser(
    description="Plot TV vs num_examples with std across seeds",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--graph", type=str, default="tree5", help="graph: tree, general, chain")
parser.add_argument("--num_examples", type=int, default=10)
args = parser.parse_args()

SEEDS = [1111, 2222 , 3333, 4444, 5555]

rows = []
for seed in SEEDS:
    file_path = Path(
        f"/home/dennis/JASACODE/runs/{args.graph}/seed_{seed}/50to500/20000_eval_tv.csv"
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

# Create 2 subplots side by side
fig, axes = plt.subplots(
    nrows=1, ncols=2,
    figsize=(14, 5),
    sharex=True, sharey=True
)

# ===== Left plot: All target indices together (same format as before) =====
ax_left = axes[0]
# First aggregate per (num_examples, target_index, method) across seeds
left_agg = (
    plot_df.groupby(["num_examples", "target_index", "method"], as_index=False)["tv"]
    .agg(["mean", "std"])
    .reset_index()
)
left_agg.columns = ["num_examples", "target_index", "method", "tv_mean", "tv_std"]

# Get unique methods for consistent colors
methods = sorted(left_agg["method"].unique())
colors = plt.cm.tab10(range(len(methods)))

# Plot all target indices, grouped by method (same format - methods as colors)
# Plot each target_index separately but use method colors
for method_idx, method in enumerate(methods):
    method_data = left_agg[left_agg["method"] == method].sort_values(["target_index", "num_examples"])
    first_target = True
    for target_idx in sorted(method_data["target_index"].unique()):
        target_data = method_data[method_data["target_index"] == target_idx].sort_values("num_examples")
        # Only label the first target_index for each method to avoid duplicate legend entries
        label = method if first_target else ""
        ax_left.plot(target_data["num_examples"], target_data["tv_mean"], 
                    marker="o", label=label, color=colors[method_idx], alpha=0.6)
        first_target = False

ax_left.set_title("All Target Indices")
ax_left.set_xlabel("Number of Examples")
ax_left.set_ylabel("TV Distance")
ax_left.grid(True, alpha=0.3)
ax_left.legend(bbox_to_anchor=(1.05, 1), loc="upper left", frameon=False)

# ===== Right plot: Average across all target indices =====
ax_right = axes[1]
# Average across target_index first (keeping method and num_examples)
avg_df = (
    plot_df.groupby(["num_examples", "method", "seed"], as_index=False)["tv"]
    .mean()
)

sns.lineplot(
    data=avg_df.sort_values("num_examples"),
    x="num_examples",
    y="tv",
    hue="method",
    estimator="mean",   # mean across seeds
    errorbar="sd",      # std across seeds
    marker="o",
    ax=ax_right
)

ax_right.set_title("Averaged Across All Target Indices")
ax_right.set_xlabel("Number of Examples")
ax_right.set_ylabel("TV Distance")
ax_right.grid(True, alpha=0.3)
ax_right.legend(bbox_to_anchor=(1.05, 1), loc="upper left", frameon=False)

fig.tight_layout()
out_path = f"/home/dennis/JASACODE/runs/{args.graph}/num_examples_{args.num_examples}_tv.png"
plt.savefig(out_path, dpi=640)
print(f"Saved: {out_path}")