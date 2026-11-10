"""
Re-run the synthetic Sachs evaluation of train_sachs.py from a saved checkpoint.

Same test graphs and episodes as at training time (they depend only on --seed,
--test-size and --cpt-dirichlet-alpha). The CSV now also carries KL columns and
the full probability vectors.

Usage:
  python eval_sachs_synth.py --ckpt-path .../best.ckpt --seed 1111 --output-csv out/x_eval_tv.csv
"""
from __future__ import annotations

import argparse
import os

import torch

from train_sachs import evaluate
from utils import ICLLightningModuleCategorical


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-path", required=True)
    ap.add_argument("--seed", type=int, required=True, help="training seed of the checkpoint")
    ap.add_argument("--test-size", type=int, default=1000)
    ap.add_argument("--cpt-dirichlet-alpha", type=float, default=0.1)
    ap.add_argument("--output-csv", required=True, help="must end in _eval_tv.csv")
    args = ap.parse_args()
    if not args.output_csv.endswith("_eval_tv.csv"):
        raise SystemExit("--output-csv must end in _eval_tv.csv")
    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)

    lit = ICLLightningModuleCategorical.load_from_checkpoint(args.ckpt_path, strict=False, map_location="cpu")
    model = lit.model.eval()
    print(f"Loaded {type(model).__name__} (loss={lit.hparams.get('loss', 'tv')}) from {args.ckpt_path}")
    run_dir = args.output_csv[: -len("_eval_tv.csv")]
    with torch.no_grad():
        evaluate(args, model, run_dir)
    print("wrote", args.output_csv)


if __name__ == "__main__":
    main()
