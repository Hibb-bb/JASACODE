#!/usr/bin/env python3
"""
Plot training curves from Lightning metrics.csv.

Features:
- quick_plot.py-like styling (bigger fonts, clean grid)
- plot both train/loss and train/tv (step-level when available)
- accept either explicit metrics.csv paths or a run directory; can auto-pick latest version_*
- for outputs_pred (prediction training), label loss as cross-entropy by default
"""
import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


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


def _latest_metrics_csv(run_dir: Path) -> Path:
    """
    Given a run_dir like .../logs, pick logs/version_*/metrics.csv with max version.
    Also accepts a version_* directory directly.
    """
    if run_dir.is_file() and run_dir.name == "metrics.csv":
        return run_dir

    # If user passed .../version_0, accept it.
    if run_dir.is_dir() and run_dir.name.startswith("version_"):
        p = run_dir / "metrics.csv"
        if not p.exists():
            raise FileNotFoundError(f"metrics.csv not found under: {run_dir}")
        return p

    # If user passed .../logs, find latest version_*
    if run_dir.is_dir():
        versions = sorted([p for p in run_dir.glob("version_*") if p.is_dir()])
        if versions:
            # numeric sort by suffix
            def _ver_num(p: Path) -> int:
                try:
                    return int(p.name.split("_", 1)[1])
                except Exception:
                    return -1

            latest = sorted(versions, key=_ver_num)[-1]
            p = latest / "metrics.csv"
            if not p.exists():
                raise FileNotFoundError(f"metrics.csv not found under: {latest}")
            return p

    raise FileNotFoundError(
        f"Could not resolve metrics.csv from path: {run_dir}. "
        "Pass a metrics.csv, a version_* dir, or a logs/ dir."
    )


def _extract_series(df: pd.DataFrame, step_col: str, value_col: str) -> Tuple[np.ndarray, np.ndarray]:
    d = df[[step_col, value_col]].dropna()
    x = d[step_col].to_numpy()
    y = d[value_col].to_numpy()
    return x, y

def main():
    p = argparse.ArgumentParser(description="Plot training loss/TV from Lightning metrics.csv")
    p.add_argument(
        "--metrics-csv",
        nargs="*",
        default=[],
        help="One or more explicit metrics.csv paths.",
    )
    p.add_argument(
        "--run-dir",
        nargs="*",
        default=[],
        help="One or more run dirs: logs/ or logs/version_*/ or a metrics.csv path. Uses latest version_* when logs/ is given.",
    )
    p.add_argument(
        "--label",
        nargs="*",
        default=[],
        help="Optional labels (same count as inputs). If omitted, uses filename parent folders.",
    )
    p.add_argument(
        "--loss-kind",
        type=str,
        default="auto",
        choices=["auto", "l1", "crossentropy"],
        help="Y-axis label for loss. auto => crossentropy if path contains outputs_pred, else 'Loss'.",
    )
    p.add_argument("-o", "--output", help="Save figure to path (png)")
    args = p.parse_args()

    inputs: List[Path] = []
    for s in args.metrics_csv:
        inputs.append(Path(s))
    for s in args.run_dir:
        inputs.append(Path(s))
    if not inputs:
        raise SystemExit("Provide --metrics-csv and/or --run-dir")

    metrics_paths: List[Path] = [_latest_metrics_csv(p) for p in inputs]
    labels: List[str] = []
    if args.label:
        labels = list(args.label)
        if len(labels) != len(metrics_paths):
            raise SystemExit("--label count must match number of inputs")
    else:
        for mp in metrics_paths:
            # e.g. .../logs/version_0/metrics.csv -> use 3 parents up as a short label
            parts = mp.parts
            labels.append("/".join(parts[-6:-2]) if len(parts) >= 6 else mp.parent.as_posix())

    fig, (ax_loss, ax_tv) = plt.subplots(nrows=2, ncols=1, figsize=(12, 8), sharex=True)

    for mp, lab in zip(metrics_paths, labels):
        df = pd.read_csv(mp)

        # Prefer step-level columns; fall back to epoch-level if needed.
        loss_col = None
        for c in ["train/loss_step", "train/loss_epoch", "train/loss"]:
            if c in df.columns and df[c].notna().any():
                loss_col = c
                break
        tv_col = None
        for c in ["train/tv_step", "train/tv_epoch", "train/tv"]:
            if c in df.columns and df[c].notna().any():
                tv_col = c
                break

        if loss_col is not None:
            x, y = _extract_series(df, "step", loss_col)
            ax_loss.plot(x, y, label=lab, alpha=0.85)
        else:
            print(f"WARNING: no loss column found in {mp}")

        if tv_col is not None:
            x, y = _extract_series(df, "step", tv_col)
            ax_tv.plot(x, y, label=lab, alpha=0.85)
        else:
            print(f"WARNING: no tv column found in {mp}")

    # Axis labels
    if args.loss_kind == "crossentropy":
        loss_ylabel = "Cross-entropy loss"
    elif args.loss_kind == "l1":
        loss_ylabel = "TV loss"
    else:
        # auto
        joined = " ".join(str(p) for p in metrics_paths)
        loss_ylabel = "Cross-entropy loss" if "outputs_pred" in joined else "Loss"

    ax_loss.set_ylabel(loss_ylabel)
    ax_loss.set_title("Training loss")
    ax_loss.grid(True, alpha=0.3)
    ax_loss.legend(fontsize=12, loc="upper right")

    ax_tv.set_xlabel("Step")
    ax_tv.set_ylabel("TV Distance")
    ax_tv.set_title("Training TV")
    ax_tv.grid(True, alpha=0.3)
    ax_tv.legend(fontsize=12, loc="upper right")

    plt.tight_layout()
    if args.output:
        plt.savefig(args.output, dpi=300, bbox_inches="tight")
        print(f"Saved to {args.output}")
    else:
        plt.show()

if __name__ == "__main__":
    main()