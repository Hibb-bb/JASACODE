import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FONT_SIZE = 20
plt.rcParams.update(
    {
        "font.size": FONT_SIZE,
        "axes.titlesize": FONT_SIZE,
        "axes.labelsize": FONT_SIZE,
        "xtick.labelsize": FONT_SIZE - 4,
        "ytick.labelsize": FONT_SIZE - 4,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    }
)


def pretty_treatment(name: str) -> str:
    # File stems are already treatment names (e.g. b2camp, cd3cd28, pma).
    # Make it display-friendly without changing meaning.
    return name.replace("_", " ").replace("-", " ").strip()


def flatten_cols(cols) -> List[str]:
    out = []
    for c in cols:
        if isinstance(c, tuple):
            out.append(f"{c[0]}_{c[1]}" if c[1] else str(c[0]))
        else:
            out.append(str(c))
    return out


def plot_one_treatment(treatment: str, seed_csvs: Dict[int, Path], out_path: Path) -> None:
    rows = []
    for seed, csv_path in seed_csvs.items():
        df = pd.read_csv(csv_path)
        # Aggregate per (context_len, target_index) for this seed (mean over episodes)
        seed_vals = (
            df.groupby(["context_len", "target_index"], as_index=False)[
                ["tv_model", "tv_naive", "tv_bayes"]
            ]
            .mean()
            .rename(columns={"context_len": "num_examples"})
        )
        seed_vals["seed"] = seed
        rows.append(seed_vals)

    wide_df = pd.concat(rows, ignore_index=True)

    # Per-node mean±std across seeds
    agg = (
        wide_df.groupby(["num_examples", "target_index"])[["tv_model", "tv_naive", "tv_bayes"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    agg.columns = flatten_cols(agg.columns)

    # For averaged panel: average over nodes per seed, then mean±std across seeds
    avg_per_seed = (
        wide_df.groupby(["num_examples", "seed"], as_index=False)[["tv_model", "tv_naive", "tv_bayes"]]
        .mean()
    )
    avg_agg = (
        avg_per_seed.groupby("num_examples")[["tv_model", "tv_naive", "tv_bayes"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    avg_agg.columns = flatten_cols(avg_agg.columns)

    target_indices = sorted(agg["target_index"].unique())
    n_targets = len(target_indices)
    target_colors = plt.cm.tab10(np.arange(n_targets))

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 9), sharex=True, sharey=True)

    def plot_panel(ax, df_agg, mean_col, std_col, title, show_legend):
        for i, t in enumerate(target_indices):
            sub = df_agg[df_agg["target_index"] == t].sort_values("num_examples")
            x = sub["num_examples"].to_numpy()
            y = sub[mean_col].to_numpy()
            yerr = sub[std_col].to_numpy()
            if np.any(np.isnan(yerr)):
                yerr = None
            ax.errorbar(
                x,
                y,
                yerr=yerr,
                marker="o",
                color=target_colors[i],
                alpha=0.7,
                capsize=3,
                capthick=1,
                label=f"Node {t}" if show_legend else None,
            )
        ax.set_title(title)
        ax.set_xlabel("Number of Examples")
        ax.set_ylabel("TV Distance")
        ax.grid(True, alpha=0.3)
        if show_legend:
            ax.legend(fontsize=14, loc="upper right")

    # (a) Transformer
    plot_panel(axes[0, 0], agg, "tv_model_mean", "tv_model_std", "Transformer", show_legend=True)
    axes[0, 0].set_xlabel("")

    # (d) Bayes
    plot_panel(axes[0, 1], agg, "tv_bayes_mean", "tv_bayes_std", "Bayesian Inference", show_legend=False)
    axes[0, 1].set_xlabel("")
    axes[0, 1].set_ylabel("")

    # (c) Naive
    plot_panel(axes[1, 0], agg, "tv_naive_mean", "tv_naive_std", "Naive Inference", show_legend=False)

    # (b) Average across nodes
    ax_br = axes[1, 1]
    for col, label in [("tv_model", "Transformer"), ("tv_naive", "Naive"), ("tv_bayes", "Bayes")]:
        sub = avg_agg.sort_values("num_examples")
        x = sub["num_examples"].to_numpy()
        y = sub[f"{col}_mean"].to_numpy()
        yerr = sub[f"{col}_std"].to_numpy()
        if np.any(np.isnan(yerr)):
            yerr = None
        ax_br.errorbar(x, y, yerr=yerr, marker="o", label=label, capsize=3, capthick=1)
    ax_br.set_title("Averaged Across Nodes")
    ax_br.set_xlabel("Number of Examples")
    ax_br.set_ylabel("")
    ax_br.grid(True, alpha=0.3)
    ax_br.legend(fontsize=14, loc="upper right")

    # Panel labels
    for ax, lab in zip(axes.flat, ["(a)", "(b)", "(c)", "(d)"]):
        ax.text(
            0.02,
            0.98,
            lab,
            transform=ax.transAxes,
            fontsize=FONT_SIZE,
            fontweight="bold",
            va="top",
            ha="left",
        )

    fig.suptitle(f"Sachs real data: {pretty_treatment(treatment)}", fontsize=FONT_SIZE + 6, y=0.90)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Aggregate eval_sachs_real across seeds (quick_plot style), one figure per treatment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--base-dir",
        type=str,
        default="runs/sachs_real_eval",
        help="Base directory containing per-seed eval dirs (base/seed_1111, ...).",
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default="imgs/sachs_real_avg5",
        help="Directory to write aggregated figures.",
    )
    ap.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[1111, 2222, 3333, 4444, 5555],
        help="Seeds to aggregate.",
    )
    args = ap.parse_args()

    base = Path(args.base_dir)
    out_dir = Path(args.out_dir)

    # Discover treatments by looking at seed_*/ *_eval_tv_real.csv
    treatments = set()
    per_seed_files: Dict[int, Dict[str, Path]] = {}
    for seed in args.seeds:
        seed_dir = base / f"seed_{seed}"
        files = list(seed_dir.glob("*_eval_tv_real.csv"))
        seed_map: Dict[str, Path] = {}
        for f in files:
            stem = f.name.replace("_eval_tv_real.csv", "")
            seed_map[stem] = f
            treatments.add(stem)
        per_seed_files[seed] = seed_map

    if not treatments:
        raise FileNotFoundError(f"No *_eval_tv_real.csv files found under {base}")

    treatments = sorted(treatments)
    print(f"Found {len(treatments)} treatments: {treatments}")

    for tr in treatments:
        seed_csvs: Dict[int, Path] = {}
        missing = []
        for seed in args.seeds:
            f = per_seed_files.get(seed, {}).get(tr)
            if f is None or not f.is_file():
                missing.append(seed)
            else:
                seed_csvs[seed] = f
        if missing:
            print(f"WARNING: treatment {tr} missing for seeds {missing}; skipping this treatment.")
            continue

        out_path = out_dir / f"{tr}_avg{len(args.seeds)}.png"
        print(f"Plotting {tr} -> {out_path}")
        plot_one_treatment(tr, seed_csvs, out_path)


if __name__ == "__main__":
    main()

