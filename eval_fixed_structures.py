#!/usr/bin/env python3
"""
Evaluate a trained model on fixed structures (tree, chain, general)
with the paper-correct naive baseline (graph-agnostic FC MLE)
and the known-DAG Bayesian baseline.
"""
from __future__ import annotations

import argparse
import os
from typing import Optional

import torch
import numpy as np

from data import (
    compile_template_from_structure,
    init_graph_params_uniform,
    get_mixed_graph_structures_5node,
    get_mixed_graph_structures,
    get_mixed_graph_structures_10node,
    get_structure_names,
)
from utils import evaluate_tv_over_context_with_baselines, EvalSpec
from utils.trainer import ICLLightningModule


def get_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate model on fixed structures with corrected baselines",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--test-size", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=9999)
    parser.add_argument("--context-lens", type=int, nargs="+",
                        default=[1, 2, 5, 10, 20, 50, 100, 200, 300, 400, 500])
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-nodes", type=int, default=5, choices=[5, 7, 10])
    return parser.parse_args(argv)


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading checkpoint: {args.checkpoint}")
    lit = ICLLightningModule.load_from_checkpoint(args.checkpoint, map_location=device)
    model = lit.model.to(device).eval()
    print(f"  layers={lit.hparams.n_layer}  embd={lit.hparams.n_embd}  heads={lit.hparams.n_head}")

    if args.num_nodes == 5:
        structures = get_mixed_graph_structures_5node(seed=42)
    elif args.num_nodes == 7:
        structures = get_mixed_graph_structures(seed=42)
    else:
        structures = get_mixed_graph_structures_10node(seed=42)
    structure_names = get_structure_names()

    param_rng = np.random.default_rng(args.seed)

    for i, (bn, name) in enumerate(zip(structures, structure_names)):
        template = compile_template_from_structure(bn)
        print(f"\n{'='*60}")
        print(f"Evaluating: {name}  (topo={template.topo_nodes})")

        p1_list = init_graph_params_uniform(
            template,
            num_graphs=args.test_size,
            seed=int(param_rng.integers(0, 1_000_000)),
        )

        output_csv = os.path.join(args.output_dir, f"eval_tv_{name}.csv")
        spec = EvalSpec(
            context_lens=args.context_lens,
            num_episodes=args.test_size,
            seed=123 + i * 1000,
            output_csv=output_csv,
            device=device,
            infer_batch_size=args.batch_size,
        )

        evaluate_tv_over_context_with_baselines(model, template, p1_list, spec)
        print(f"  -> {output_csv}")

    print(f"\nDone. Results in {args.output_dir}")


if __name__ == "__main__":
    main()
