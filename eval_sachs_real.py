from __future__ import annotations

import argparse
import os
from glob import glob
from typing import List

import numpy as np
import pandas as pd
import torch

from data import get_sachs, compile_template_from_categorical
from utils import ICLLightningModuleCategorical, EvalSpec
from utils.sachs_real_eval import (
    encode_disc_df_to_int,
    empirical_cpt_from_data,
    evaluate_tv_over_context_categorical_real,
)
from train_sachs import plot_results


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate trained Sachs transformer on real Sachs discretized data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--ckpt-path",
        type=str,
        required=True,
        help="Path to trained ICLLightningModuleCategorical checkpoint (best.ckpt).",
    )
    parser.add_argument(
        "--disc-data-dir",
        type=str,
        default="Sachs/disc_data",
        help="Directory containing discretized Sachs CSVs (one per treatment).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/sachs_real",
        help="Base directory for saving evaluation CSVs and plots.",
    )
    parser.add_argument(
        "--context-lens",
        type=int,
        nargs="+",
        default=[1, 2, 5, 10, 20, 50, 100, 200, 300, 400, 500],
        help="Context lengths (number of context examples) to evaluate.",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=2000,
        help="Number of episodes per (context_len, target_index) pair.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed for evaluation episode sampling.",
    )

    return parser.parse_args()


def main() -> None:
    args = get_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1) Template: hard-coded Sachs DAG (K=3); compile uses structure only (CPTs in bn unused here).
    print("Loading Sachs categorical template (hard-coded DAG, K=3)...")
    bn = get_sachs(seed=2000)
    template = compile_template_from_categorical(bn)
    node_order: List[str] = list(template.topo_nodes)
    print(f"Template nodes (topological order): {node_order}")

    # 2) Load trained transformer from checkpoint.
    if not os.path.exists(args.ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt_path}")

    print(f"Loading checkpoint: {args.ckpt_path}")
    lit = ICLLightningModuleCategorical.load_from_checkpoint(
        args.ckpt_path,
        strict=False,
    )
    model = lit.model
    model.eval()

    # 3) Discover discretized Sachs treatment CSVs.
    csv_paths = sorted(glob(os.path.join(args.disc_data_dir, "*.csv")))
    if not csv_paths:
        raise FileNotFoundError(
            f"No CSV files found in discretized data directory: {args.disc_data_dir}"
        )

    print(f"Found {len(csv_paths)} treatment CSVs in {args.disc_data_dir}")

    os.makedirs(args.output_dir, exist_ok=True)

    for csv_path in csv_paths:
        treatment_name = os.path.splitext(os.path.basename(csv_path))[0]
        print(f"\n=== Evaluating treatment: {treatment_name} ===")
        print(f"Reading: {csv_path}")

        df_raw = pd.read_csv(csv_path)

        # Some discretized Sachs CSVs use slightly different capitalization (e.g. "Raf" vs "RAF").
        # Align columns to the template node names using a case-insensitive match when needed.
        if any(c not in df_raw.columns for c in node_order):
            lower_to_actual = {c.lower(): c for c in df_raw.columns}
            rename_map = {}
            for expected in node_order:
                if expected in df_raw.columns:
                    continue
                actual = lower_to_actual.get(expected.lower())
                if actual is not None:
                    rename_map[actual] = expected
            if rename_map:
                df_raw = df_raw.rename(columns=rename_map)

        # Reorder columns to match template topo order.
        missing = [c for c in node_order if c not in df_raw.columns]
        if missing:
            raise ValueError(
                f"Columns {missing} from template not found in {csv_path}. "
                "Ensure the discretized Sachs CSVs use the same variable names."
            )
        df = df_raw[node_order].copy()

        # Encode interval strings into integer codes 0..K-1.
        X_data, _ = encode_disc_df_to_int(df, expected_cardinality=template.cardinality)
        print(f"Encoded real data shape for {treatment_name}: {X_data.shape}")

        # Build empirical CPTs from the full treatment dataset.
        print("Computing empirical CPTs from real data...")
        cpt_emp_list = empirical_cpt_from_data(X_data, template)

        # Prepare EvalSpec for this treatment.
        out_csv = os.path.join(
            args.output_dir,
            f"{treatment_name}_eval_tv_real.csv",
        )
        spec = EvalSpec(
            context_lens=args.context_lens,
            num_episodes=args.num_episodes,
            seed=args.seed,
            output_csv=out_csv,
            device=device,
            infer_batch_size=16,
        )

        print("Running real-data evaluation episodes...")
        evaluate_tv_over_context_categorical_real(
            model=model,
            template=template,
            X_data=X_data,
            cpt_emp_list=cpt_emp_list,
            spec=spec,
            treatment_name=treatment_name,
        )
        print(f"Wrote evaluation CSV: {out_csv}")

        # Generate per-treatment plot using existing plotting helper.
        plot_output_path = os.path.join(
            args.output_dir,
            f"{treatment_name}_eval_tv_real.png",
        )
        print("Generating TV distance plot...")
        # Reuse train_sachs.plot_results; args is only used for labels, so pass through.
        plot_results(args, out_csv, plot_output_path)
        print(f"Saved plot: {plot_output_path}")


if __name__ == "__main__":
    main()

