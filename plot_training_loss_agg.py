#!/usr/bin/env python3
"""Aggregate training loss across seeds (mean ± std) and save a single plot."""
import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", required=True,
                   help="Top-level run directory containing seed subdirectories")
    p.add_argument("--title", default="Training Loss (aggregated over seeds)")
    p.add_argument("--smoothing", type=int, default=50,
                   help="Rolling-window size for smoothing (0 = no smoothing)")
    args = p.parse_args()

    seed_dirs = sorted([
        os.path.join(args.output_dir, d)
        for d in os.listdir(args.output_dir)
        if os.path.isdir(os.path.join(args.output_dir, d))
        and not d.endswith("_eval")
        and not d.endswith(".png")
    ])

    all_series = []
    for sd in seed_dirs:
        csv_path = os.path.join(sd, "logs", "version_0", "metrics.csv")
        if not os.path.isfile(csv_path):
            print(f"  [skip] {csv_path} not found")
            continue
        df = pd.read_csv(csv_path)
        if "train/loss_step" not in df.columns:
            print(f"  [skip] train/loss_step not in {csv_path}")
            continue
        sub = df[["step", "train/loss_step"]].dropna()
        sub = sub.sort_values("step").reset_index(drop=True)
        all_series.append(sub)
        print(f"  Loaded {len(sub)} rows from {os.path.basename(sd)}")

    if not all_series:
        print("No data found — nothing to plot.")
        return

    merged = all_series[0][["step"]].rename(columns={"train/loss_step": "loss_0"})
    for i, s in enumerate(all_series):
        s = s.rename(columns={"train/loss_step": f"loss_{i}"})
        if i == 0:
            merged = s
        else:
            merged = pd.merge(merged, s, on="step", how="inner")

    loss_cols = [c for c in merged.columns if c.startswith("loss_")]
    steps = merged["step"].values
    loss_mat = merged[loss_cols].values  # (num_steps, num_seeds)

    mean_loss = loss_mat.mean(axis=1)
    std_loss = loss_mat.std(axis=1)

    if args.smoothing > 0 and len(mean_loss) > args.smoothing:
        kernel = np.ones(args.smoothing) / args.smoothing
        mean_smooth = np.convolve(mean_loss, kernel, mode="valid")
        std_smooth = np.convolve(std_loss, kernel, mode="valid")
        steps_smooth = steps[args.smoothing - 1:]
    else:
        mean_smooth = mean_loss
        std_smooth = std_loss
        steps_smooth = steps

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(steps_smooth, mean_smooth, color="C0", linewidth=1.5, label="Mean")
    ax.fill_between(steps_smooth,
                    mean_smooth - std_smooth,
                    mean_smooth + std_smooth,
                    alpha=0.25, color="C0", label="± 1 std")
    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Training Loss (TV)", fontsize=12)
    ax.set_title(args.title, fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = os.path.join(args.output_dir, "final_training_loss.png")
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
