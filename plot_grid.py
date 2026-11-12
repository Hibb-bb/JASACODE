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
    fig = plt.figure(figsize=(20, 24), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.0, hspace=0.0)
    sub = fig.subfigures(3, 2, hspace=0.0, wspace=0.0)
    for r, (row_label, g5, g7) in enumerate(ROWS):
        for c, g in enumerate((g5, g7)):
            key = (args.loss, g, THM_TAG[g])
            if key not in found:
                raise SystemExit(f"missing runs for {key}")
            wide = pr.load_group(found[key])
            if args.x_range:
                wide = wide[wide["num_examples"].between(*args.x_range)]
            sf = sub[r, c]
            axes = sf.subplots(2, 2, sharex=True, sharey=True)
            pr.draw_2x2(axes, wide)
            if r == 0:
                sf.suptitle(COLS[c], fontsize=pr.FONT_SIZE + 12, fontweight="bold")
            if c == 0:
                sf.supylabel(row_label, fontsize=pr.FONT_SIZE + 12, fontweight="bold")
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)
    print("saved", out)


if __name__ == "__main__":
    main()
