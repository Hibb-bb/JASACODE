"""Big grid of the per-config 2x2 blocks: rows = graph (general, chain, tree),
columns = 5-node / 7-node versions. One figure per x-range.

  python plot_grid.py --metric kl --loss bce --out figures_thm/grid/kl_ce.png
  python plot_grid.py --metric kl --loss bce --x-range 300 500 --out figures_thm/grid/kl_ce_zoom300.png
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import plot_rebuttal as pr

# (row label, 5-node graph, 7-node graph); the narrow theorem model for each.
ROWS = [("General graph", "general", "general7"),
        ("Chain", "chain5", "chain"),
        ("Tree", "tree5", "tree")]
COLS = ["5 node", "7 node"]
THM_TAG = {"general": "theory_L2H1D22", "general7": "theory_L2H1D30",
           "chain5": "theory_L2H1D22", "chain": "theory_L2H1D30",
           "tree5": "theory_L2H1D22", "tree": "theory_L2H1D30"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs_rebuttal")
    ap.add_argument("--loss", choices=["tv", "bce"], default="bce")
    ap.add_argument("--metric", choices=["tv", "kl"], default="kl")
    ap.add_argument("--x-range", type=int, nargs=2, default=None)
    ap.add_argument("--log-y", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    pr.METRIC["log_y"] = args.log_y
    if args.metric == "kl":
        pr.METRIC.update(name="kl", ylabel="KL Divergence")

    found = pr.discover(Path(args.root))
    plt.rcParams.update(pr.PLOT_RC)
    fig = plt.figure(figsize=(21, 25), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.0, hspace=0.0)
    # Narrow first column holds the row labels so they never collide with y-axis labels.
    sub = fig.subfigures(3, 3, width_ratios=[0.05, 1, 1], hspace=0.025, wspace=0.0)
    for r, (row_label, g5, g7) in enumerate(ROWS):
        sub[r, 0].text(0.5, 0.5, row_label, rotation=90, ha="center", va="center",
                       fontsize=pr.FONT_SIZE + 12, fontweight="bold")
        for c, g in enumerate((g5, g7)):
            key = (args.loss, g, THM_TAG[g])
            if key not in found:
                raise SystemExit(f"missing runs for {key}")
            wide = pr.load_group(found[key])
            if args.x_range:
                wide = wide[wide["num_examples"].between(*args.x_range)]
            sf = sub[r, c + 1]
            axes = sf.subplots(2, 2, sharex=True, sharey=True)
            pr.draw_2x2(axes, wide)
            if c == 1:  # drop the y-axis label on the right column but keep its space
                for a in axes[:, 0]:
                    a.set_ylabel(" ")
            if r == 0:
                sf.suptitle(COLS[c], fontsize=pr.FONT_SIZE + 12, fontweight="bold")
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)
    print("saved", out)


if __name__ == "__main__":
    main()
