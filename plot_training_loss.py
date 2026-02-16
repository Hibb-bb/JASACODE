#!/usr/bin/env python3
"""Quick plot of training loss from Lightning metrics.csv."""
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def main():
    p = argparse.ArgumentParser(description="Plot training loss from metrics.csv")
    p.add_argument("--metrics_csv", nargs="?", default="runs/tree/seed_1234/200/20000/logs/version_0/metrics.csv",
                   help="Path to metrics.csv")
    p.add_argument("-o", "--output", help="Save figure to path")
    args = p.parse_args()

    df = pd.read_csv(args.metrics_csv)

    # Use step-wise loss where available, else epoch loss
    has_step = "train/loss_step" in df.columns and df["train/loss_step"].notna().any()
    # has_epoch = "train/loss_epoch" in df.columns and df["train/loss_epoch"].notna().any()

    fig, ax = plt.subplots()
    if has_step:
        step = df["step"].dropna()
        loss_step = df["train/loss_step"].dropna()
        if len(step) != len(loss_step):
            step = step.iloc[: len(loss_step)]
        ax.plot(step, loss_step, label="train/loss_step", color="C0")
    # if has_epoch:
    #     epoch_df = df[df["train/loss_epoch"].notna()]
    #     if not epoch_df.empty:
    #         ax.scatter(epoch_df["step"], epoch_df["train/loss_epoch"], label="train/loss_epoch", color="C1", zorder=5)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if args.output:
        plt.savefig(args.output, dpi=150)
        print(f"Saved to {args.output}")
    else:
        plt.show()

if __name__ == "__main__":
    main()
