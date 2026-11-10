"""Per-term training-loss curves for papers_replication.py runs.

Reads <run_dir>/loss_curves/<variant>_s<seed>.csv and writes <variant>.png.
  python plot_loss_curves.py runs_papers/clean runs_papers/realistic [--linear-x]
"""
import argparse, glob, os, re
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def smooth_log(y, frac=0.05):
    """Mean over a window that grows with the step (about frac * step wide),
    so early steps stay sharp and late steps are smoothed, as on a log axis."""
    y = np.asarray(y, float)
    cs = np.concatenate([[0.0], np.cumsum(y)])
    n = len(y); i = np.arange(n)
    h = np.maximum(0, (frac * (i + 1) / 2).astype(int))
    lo, hi = np.maximum(0, i - h), np.minimum(n, i + h + 1)
    return (cs[hi] - cs[lo]) / (hi - lo)


def plot_variant(run_dir, tag, title=None, logx=True):
    files = sorted(glob.glob(os.path.join(run_dir, "loss_curves", f"{tag}_s*.csv")))
    dfs = {}
    for f in files:
        m = re.fullmatch(re.escape(tag) + r"_s(\d+)\.csv", os.path.basename(f))
        if m:
            dfs[int(m.group(1))] = pd.read_csv(f)
    if not dfs:
        return None
    terms = [c for c in next(iter(dfs.values())).columns if c not in ("step", "npairs", "nsub")]
    nc = 3; nr = int(np.ceil(len(terms) / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(4.2 * nc, 3.0 * nr), squeeze=False)
    for ax, term in zip(axes.flat, terms):
        for seed, df in sorted(dfs.items()):
            x = df["step"].to_numpy() + (1 if logx else 0)
            y = df[term].to_numpy()
            ys = smooth_log(y) if logx else pd.Series(y).rolling(max(1, len(y) // 100), min_periods=1).mean()
            ax.plot(x, y, alpha=0.15, lw=0.5, color=f"C{seed % 10}")
            ax.plot(x, ys, lw=1.2, color=f"C{seed % 10}", label=f"seed {seed}")
        allv = pd.concat([d[term] for d in dfs.values()])
        if (allv > 0).all():
            ax.set_yscale("log")
        if logx:
            ax.set_xscale("log")
        ax.set_title(term); ax.set_xlabel("step + 1" if logx else "step"); ax.grid(alpha=0.3, which="both" if logx else "major")
    for ax in list(axes.flat)[len(terms):]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=7)
    fig.suptitle(title or tag)
    fig.tight_layout()
    out = os.path.join(run_dir, "loss_curves", f"{tag}.png")
    fig.savefig(out, dpi=130); plt.close(fig)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--linear-x", action="store_true")
    a = ap.parse_args()
    for d in a.run_dirs:
        tags = sorted({re.sub(r"_s\d+\.csv$", "", os.path.basename(f))
                       for f in glob.glob(os.path.join(d, "loss_curves", "*_s*.csv"))})
        for tag in tags:
            print(plot_variant(d, tag, title=f"{tag.replace('_', '+')}  ({os.path.basename(os.path.normpath(d))})", logx=not a.linear_x))
