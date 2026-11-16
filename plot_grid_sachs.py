"""2-column x 3-row grid of Sachs per-treatment 2x2 blocks; each cell is titled by its condition.

  python plot_grid_sachs.py --metric kl --out figures_thm/grid/sachs_kl_ce.png
  python plot_grid_sachs.py --metric kl --x-range 300 500 --out figures_thm/grid/sachs_kl_ce_zoom300.png
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import quick_plot_sachs as qs

# (file stem, display name), filled row by row.
CONDITIONS = [
    ("cd3cd28", "CD3/CD28"),
    ("cd3cd28_g0076", "CD3/CD28 + G0076"),
    ("cd3cd28_ly", "CD3/CD28 + LY294002"),
    ("cd3cd28_icam2", "CD3/CD28 + ICAM-2"),
    ("cd3cd28_aktinhib", "CD3/CD28 + AKT inhibitor"),
    ("cd3cd28_u0126", "CD3/CD28 + U0126"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", default="runs_sachs_thm/kl/real_theory_L2H1D69_ce_ts5000")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1111, 2222, 3333, 4444, 5555])
    ap.add_argument("--metric", choices=["tv", "kl"], default="kl")
    ap.add_argument("--x-range", type=int, nargs=2, default=None)
    ap.add_argument("--log-y", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    qs.METRIC["log_y"] = args.log_y
    qs.METRIC["x_range"] = args.x_range
    if args.metric == "kl":
        qs.METRIC.update(name="kl", ylabel="KL Divergence")

    base = Path(args.base_dir)
    fig = plt.figure(figsize=(21, 25), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.0, hspace=0.0)
    sub = fig.subfigures(3, 2, hspace=0.025, wspace=0.0)
    for i, (stem, name) in enumerate(CONDITIONS):
        r, c = divmod(i, 2)
        seed_csvs = {s: base / f"seed_{s}" / f"{stem}_eval_tv_real.csv" for s in args.seeds}
        missing = [s for s, p in seed_csvs.items() if not p.is_file()]
        if missing:
            raise SystemExit(f"{stem}: missing seeds {missing} under {base}")
        sf = sub[r, c]
        axes = sf.subplots(2, 2, sharex=True, sharey=True)
        qs.draw_treatment(axes, seed_csvs)
        if c == 1:  # drop the y-axis label on the right column but keep its space
            for a in axes[:, 0]:
                a.set_ylabel(" ")
        sf.suptitle(name, fontsize=qs.FONT_SIZE + 4, fontweight="bold")
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)
    print("saved", out)


if __name__ == "__main__":
    main()
