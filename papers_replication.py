"""Script version of papers_replication.ipynb.

Trains and probes every variant, then runs controls.py and diag.py, exactly as
the notebook does. All outputs (checkpoints, JSON, CSV) go to --out-dir.

  python papers_replication.py --out-dir runs_papers/clean
  python papers_replication.py --out-dir runs_papers/realistic --realistic
"""
import argparse, os, runpy, shutil, sys, time, json
import numpy as np, torch, pandas as pd

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
import lcjepa
from lcjepa import *

ap = argparse.ArgumentParser()
ap.add_argument("--out-dir", required=True)
ap.add_argument("--realistic", action="store_true", help="nightly=True, realistic=True (noisy, gappy test)")
ap.add_argument("--steps", type=int, default=None, help="override STEPS (smoke tests)")
ap.add_argument("--seeds", type=str, default=None, help="override SEEDS, e.g. 0,1,2")
ap.add_argument("--variants", type=str, default=None, help="override VARIANTS, comma separated")
args = ap.parse_args()
os.makedirs(args.out_dir, exist_ok=True)
os.chdir(args.out_dir)   # checkpoints and JSON land here, like the notebook's cwd

# ---- cell 2
device = "cuda" if torch.cuda.is_available() else "cpu"
lcjepa.SIGREG_SLICES = 1024 if device == "cuda" else 256   # official LeJEPA uses 1024 slices
print("device:", device, "| SIGReg slices:", lcjepa.SIGREG_SLICES, "| loss weights:", lcjepa.LAM, flush=True)

# ---- cell 3
STEPS = 4000 if device == "cuda" else 600
B = 128 if device == "cuda" else 32
SEEDS = [0, 1, 2] if device == "cuda" else [0]
DATA_KW = dict(nightly=args.realistic, realistic=args.realistic)   # both True for the noisy, gappy test
VARIANTS = ["mae",                          # reconstruction baseline
            "rot+batch+int-noerr",          # JEPA control (LeWM-style)
            "rot+batch+int-noerr+nce",      # contrast JEPA (our best on CPU)
            "rot+batch+int-noerr+dis",      # MotionJEPA DISReg (index-neighbour patches)
            "rot+batch+int-noerr+dis2",     # improved DISReg
            "rot+batch+int-noerr+ham+dis",  # LeHami flow + DISReg
            "rot+batch+int-noerr+ham",      # LeHami flow alone
            "angle"]                        # action-angle JEPA
if args.steps: STEPS = args.steps
if args.seeds: SEEDS = [int(s) for s in args.seeds.split(",")]
if args.variants: VARIANTS = args.variants.split(",")
print("STEPS", STEPS, "B", B, "SEEDS", SEEDS, "DATA_KW", DATA_KW, flush=True)
test = make_batch(1000, rng=np.random.default_rng(123), **DATA_KW)

# ---- added: record every loss term at every step (wraps lcjepa.loss_fn, which
# train() looks up at call time; no change to training or its random streams)
_loss_fn = lcjepa.loss_fn
STEP_LOG = []
def _recording_loss_fn(*a, **k):
    loss, logs = _loss_fn(*a, **k)
    row = {kk: float(v) for kk, v in logs.items() if isinstance(v, (int, float)) and kk not in ("nsub", "npairs")}
    row["total"] = float(loss.detach())
    STEP_LOG.append(row)
    return loss, logs
lcjepa.loss_fn = _recording_loss_fn
os.makedirs("loss_curves", exist_ok=True)

def plot_loss_curves(var):
    import glob, matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tag = var.replace("+", "_")
    dfs = {int(f.split("_s")[-1][:-4]): pd.read_csv(f) for f in sorted(glob.glob(f"loss_curves/{tag}_s*.csv"))}
    if not dfs:
        return
    terms = [c for c in next(iter(dfs.values())).columns if c != "step"]
    nc = 3; nr = int(np.ceil(len(terms) / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(4.2 * nc, 3.0 * nr), squeeze=False)
    win = max(1, STEPS // 100)
    for ax, term in zip(axes.flat, terms):
        for seed, df in dfs.items():
            y = df[term]
            ax.plot(df["step"], y, alpha=0.15, lw=0.5, color=f"C{seed % 10}")
            ax.plot(df["step"], y.rolling(win, min_periods=1).mean(), lw=1.2, color=f"C{seed % 10}", label=f"seed {seed}")
        if (pd.concat([d[term] for d in dfs.values()]) > 0).all():
            ax.set_yscale("log")
        ax.set_title(term); ax.set_xlabel("step"); ax.grid(alpha=0.3)
    for ax in list(axes.flat)[len(terms):]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=7)
    fig.suptitle(f"{var}  ({'realistic' if args.realistic else 'clean'} data)")
    fig.tight_layout()
    fig.savefig(f"loss_curves/{tag}.png", dpi=130); plt.close(fig)

# ---- cell 5: train and probe
results = {}
for var in VARIANTS:
    for seed in SEEDS:
        t0 = time.time()
        STEP_LOG.clear()
        model, hist = train(var, steps=STEPS, B=B, device=device, seed=seed, data_kw=DATA_KW, log_every=max(1, STEPS // 6))
        res = evaluate(model, test, device=device)
        res["train_minutes"] = (time.time() - t0) / 60
        results[f"{var} | seed {seed}"] = res
        torch.save(model.state_dict(), f"ckpt_{var}_s{seed}.pt")
        pd.DataFrame(STEP_LOG).rename_axis("step").reset_index().to_csv(
            f"loss_curves/{var.replace('+', '_')}_s{seed}.csv", index=False)
        plot_loss_curves(var)
        json.dump(results, open("results_papers.json", "w"), indent=1)
        print("done:", var, "seed", seed, f"{res['train_minutes']:.1f} min", flush=True)

# ---- cell 7: results table
cols = {"phase | h (token hidden) + c | MLP R2": "phase h+c", "DRW state | h (token hidden) | linear R2": "DRW h",
        "log P | c+meanh | linear R2": "logP", "class | c+meanh | accuracy": "class", "train_minutes": "min"}
rows = {}
for k, r in results.items():
    rows.setdefault(k.split(" | seed ")[0], []).append([r.get(c, np.nan) for c in cols])
mean = pd.DataFrame({v: np.nanmean(np.array(rs), 0) for v, rs in rows.items()}, index=list(cols.values())).T.round(3)
std = pd.DataFrame({v: np.nanstd(np.array(rs), 0) for v, rs in rows.items()}, index=list(cols.values())).T.round(3)
pd.set_option("display.width", 200)
print("std over seeds:\n", std, "\nmean over seeds:\n", mean, flush=True)
mean.to_csv("results_mean.csv"); std.to_csv("results_std.csv")

# ---- cell 10: controls (uses seed-0 checkpoints named ckpt_<var>.pt)
s0 = SEEDS[0]
for var in VARIANTS:
    shutil.copy(f"ckpt_{var}_s{s0}.pt", f"ckpt_{var}.pt")
sys.argv = ["controls.py", ",".join(VARIANTS)]
runpy.run_path(os.path.join(REPO, "controls.py"), run_name="__main__")

# ---- cell 13: diagnostics
rot = [v for v in VARIANTS if v.startswith("rot")]
if rot:
    sys.argv = ["diag.py", ",".join(rot)]
    runpy.run_path(os.path.join(REPO, "diag.py"), run_name="__main__")
print("all done", flush=True)
