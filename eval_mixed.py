"""
Evaluation script for mixed graph structure models.

This script evaluates a model trained on mixed structures by testing it on
each structure separately. This tests whether the model can:
1. In-context learn different graph structures
2. In-context learn different CPT parameters
3. Adapt its predictions based on the observed context
"""
from __future__ import annotations

import argparse
from typing import Optional
import os
from pathlib import Path

import torch
import numpy as np

from data import (
    compile_template_from_structure,
    init_graph_params_uniform,
    get_mixed_graph_structures,
    get_structure_names,
)
from utils import evaluate_tv_over_context, EvalSpec
from utils.trainer import ICLLightningModule


def get_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate mixed structure model on each structure separately",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained model checkpoint (.ckpt file).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="eval_results_mixed",
        help="Directory to save evaluation CSVs.",
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=1000,
        help="Number of evaluation episodes per structure.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed for evaluation.",
    )
    parser.add_argument(
        "--context-lens",
        type=int,
        nargs="+",
        default=[1, 2, 5, 10, 20, 50, 100, 200, 300, 400, 500],
        help="Context lengths to evaluate.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Batch size for inference.",
    )

    args = parser.parse_args(argv)
    return args


def load_model_from_checkpoint(checkpoint_path: str, device: str = "cuda") -> torch.nn.Module:
    """Load the trained model from checkpoint."""
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # Load Lightning module
    lit_module = ICLLightningModule.load_from_checkpoint(
        checkpoint_path,
        map_location=device,
    )
    
    # Extract the actual model
    model = lit_module.model
    model.eval()
    model.to(device)
    
    print(f"Model loaded successfully")
    print(f"  Input dim: {lit_module.hparams.input_dim}")
    print(f"  Embedding dim: {lit_module.hparams.n_embd}")
    print(f"  Layers: {lit_module.hparams.n_layer}")
    print(f"  Heads: {lit_module.hparams.n_head}")
    
    return model


def main():
    args = get_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load trained model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model_from_checkpoint(args.checkpoint, device=device)
    
    # Get all structures
    print("\nLoading graph structures...")
    structures = get_mixed_graph_structures(seed=42)
    structure_names = get_structure_names()
    
    print(f"Found {len(structures)} structures: {structure_names}")
    
    # Evaluate on each structure separately
    for i, (bn, name) in enumerate(zip(structures, structure_names)):
        print("\n" + "=" * 70)
        print(f"Evaluating on structure: {name} (structure {i})")
        print("=" * 70)
        
        # Compile template
        template = compile_template_from_structure(bn)
        print(f"  Nodes: {template.num_nodes}")
        print(f"  Topo order: {template.topo_nodes}")
        
        # Initialize a SINGLE fixed graph instance for evaluation
        # This tests whether the model can learn the CPT from context
        param_rng = np.random.default_rng(args.seed + i)
        p1_list = init_graph_params_uniform(
            template, 
            num_graphs=1,  # Single fixed graph
            seed=param_rng
        )
        
        print(f"  Initialized 1 test graph with random CPT")
        
        # Evaluate across different context lengths
        output_csv = output_dir / f"eval_tv_{name}.csv"
        
        eval_spec = EvalSpec(
            context_lens=args.context_lens,
            num_episodes=args.test_size,
            seed=args.seed + i * 1000,  # Different seed per structure
            output_csv=str(output_csv),
            device=device,
            infer_batch_size=args.batch_size,
        )
        
        print(f"  Evaluating on context lengths: {args.context_lens}")
        print(f"  Episodes per context length: {args.test_size}")
        print(f"  Output CSV: {output_csv}")
        
        evaluate_tv_over_context(model, template, p1_list, eval_spec)
        
        print(f"✓ Completed evaluation on {name}")
    
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    print(f"\nResults saved to: {output_dir}")
    print("\nFiles generated:")
    for name in structure_names:
        csv_path = output_dir / f"eval_tv_{name}.csv"
        if csv_path.exists():
            print(f"  - {csv_path}")
    
    print("\nInterpretation:")
    print("  - Low TV (total variation) means good prediction")
    print("  - Compare TV across structures to see if model adapts")
    print("  - TV should decrease with more context (better in-context learning)")
    print("  - If model learned all structures, TV should be low for all")


if __name__ == "__main__":
    main()
