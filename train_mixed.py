"""
Training script for mixed graph structures.

This script trains a transformer on data from multiple graph structures simultaneously.
All structures use 5 nodes to maintain consistent input dimensions.
"""
from __future__ import annotations

import argparse
from typing import Optional
import os

import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from data import (
    compile_template_from_structure,
    init_graph_params_uniform,
    get_mixed_graph_structures,
    get_structure_names,
    MixedICLBatchSpec,
    MixedGraphICLSequenceDataset,
)
from utils import ICLLightningModule, evaluate_tv_over_context_with_baselines, EvalSpec


def get_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train transformer on mixed graph structures",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data / batch
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Total number of graphs per batch (B), split across structures.",
    )
    parser.add_argument(
        "--context-len",
        type=int,
        default=None,
        help="Fixed number of context examples per sequence. If None, use --min-context-len and --max-context-len for dynamic sampling.",
    )
    parser.add_argument(
        "--min-context-len",
        type=int,
        default=5,
        help="Minimum context length for dynamic sampling (only used if --context-len is None).",
    )
    parser.add_argument(
        "--max-context-len",
        type=int,
        default=200,
        help="Maximum context length for dynamic sampling (only used if --context-len is None).",
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=20000,
        help="Number of graph instances per structure for training.",
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=5000,
        help="Number of graph instances per structure for testing.",
    )
    parser.add_argument(
        "--target-index",
        type=int,
        default=4,
        help="Target node index (0-4 for 5-node graphs).",
    )

    # Training
    parser.add_argument(
        "--train-step",
        type=int,
        default=100_000,
        help="Number of training steps.",
    )
    parser.add_argument(
        "--init-lr",
        type=float,
        default=3e-4,
        help="Initial learning rate.",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=1000,
        help="Number of warmup steps for learning rate scheduler.",
    )
    parser.add_argument(
        "--min-lr",
        type=float,
        default=0.0,
        help="Minimum learning rate for cosine decay (default: 0.0).",
    )

    # Output / misc
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs_mixed",
        help="Directory to save checkpoints and logs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed.",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from.",
    )

    args = parser.parse_args(argv)
    return args


def evaluate_on_structures(args, model, templates, structure_names, run_dir):
    """Evaluate the model separately on each structure with baselines."""
    print("\n" + "=" * 70)
    print("EVALUATING ON EACH STRUCTURE")
    print("=" * 70)
    
    eval_dir = run_dir + "_eval"
    os.makedirs(eval_dir, exist_ok=True)
    
    # Use different seed for test graphs (like train.py)
    param_rng = np.random.default_rng(args.seed + 1000)
    
    for i, (template, name) in enumerate(zip(templates, structure_names)):
        print(f"\nEvaluating on structure: {name}")
        print(f"  Nodes: {template.num_nodes}")
        print(f"  Generating {args.test_size} test graphs...")
        
        # Generate test graphs with different CPTs
        p1_list = init_graph_params_uniform(
            template, 
            num_graphs=args.test_size,
            seed=param_rng.integers(0, 1000000)
        )
        
        output_csv = os.path.join(eval_dir, f"eval_tv_{name}.csv")
        
        eval_spec = EvalSpec(
            context_lens=[1, 2, 5, 10, 20, 50, 100, 200, 300, 400, 500],
            num_episodes=args.test_size,
            seed=123,
            output_csv=output_csv,
            device="cuda" if torch.cuda.is_available() else "cpu",
            infer_batch_size=512,
        )
        
        print(f"  Evaluating with baselines...")
        evaluate_tv_over_context_with_baselines(model, template, p1_list, eval_spec)
        print(f"  ✓ Results saved to: {output_csv}")
    
    return eval_dir


def plot_mixed_results(eval_dir, structure_names, output_path):
    """
    Generate plots comparing all structures with baselines.
    Creates a grid showing model, naive baseline, and Bayesian baseline for each structure.
    """
    print("\n" + "=" * 70)
    print("GENERATING PLOTS")
    print("=" * 70)
    
    # Load data for all structures
    all_data = {}
    for name in structure_names:
        csv_path = os.path.join(eval_dir, f"eval_tv_{name}.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            # Aggregate per (context_len, target_index)
            plot_df = (
                df.groupby(["context_len", "target_index"], as_index=False)[
                    ["tv_model", "tv_naive", "tv_bayes"]
                ]
                .mean()
                .rename(columns={"context_len": "num_examples"})
            )
            all_data[name] = plot_df
    
    if not all_data:
        print("No evaluation data found!")
        return
    
    # Create 3x3 grid: rows = structures, columns = (model, naive, bayes)
    n_structures = len(structure_names)
    fig, axes = plt.subplots(
        nrows=n_structures, ncols=3,
        figsize=(15, 5 * n_structures),
        sharex=True, sharey=True
    )
    
    if n_structures == 1:
        axes = axes.reshape(1, -1)
    
    for i, name in enumerate(structure_names):
        if name not in all_data:
            continue
        
        plot_df = all_data[name]
        target_indices = sorted(plot_df["target_index"].unique())
        n_targets = len(target_indices)
        colors = plt.cm.tab10(range(n_targets))
        
        # Column 0: Transformer
        ax_model = axes[i, 0]
        for target_idx in target_indices:
            target_data = plot_df[plot_df["target_index"] == target_idx].sort_values("num_examples")
            ax_model.plot(target_data["num_examples"], target_data["tv_model"], 
                         marker="o", label=f"Target {target_idx}", 
                         color=colors[target_indices.index(target_idx)], alpha=0.7)
        ax_model.set_title(f"{name.capitalize()} - Transformer")
        ax_model.set_ylabel("TV Distance")
        ax_model.grid(True, alpha=0.3)
        if i == 0:  # Only show legend on first row
            ax_model.legend(fontsize=8)
        
        # Column 1: Naive Baseline
        ax_naive = axes[i, 1]
        for target_idx in target_indices:
            target_data = plot_df[plot_df["target_index"] == target_idx].sort_values("num_examples")
            ax_naive.plot(target_data["num_examples"], target_data["tv_naive"], 
                         marker="o", label=f"Target {target_idx}", 
                         color=colors[target_indices.index(target_idx)], alpha=0.7)
        ax_naive.set_title(f"{name.capitalize()} - Naive Baseline")
        ax_naive.grid(True, alpha=0.3)
        
        # Column 2: Bayesian Baseline
        ax_bayes = axes[i, 2]
        for target_idx in target_indices:
            target_data = plot_df[plot_df["target_index"] == target_idx].sort_values("num_examples")
            ax_bayes.plot(target_data["num_examples"], target_data["tv_bayes"], 
                         marker="o", label=f"Target {target_idx}", 
                         color=colors[target_indices.index(target_idx)], alpha=0.7)
        ax_bayes.set_title(f"{name.capitalize()} - Bayesian (known DAG)")
        ax_bayes.grid(True, alpha=0.3)
    
    # Set x-labels only on bottom row
    for j in range(3):
        axes[-1, j].set_xlabel("Number of Examples")
    
    fig.suptitle("Mixed Graph Structure Training - Evaluation Results", 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Plot saved to: {output_path}")
    
    # Also create a comparison plot (average across all targets)
    fig2, ax = plt.subplots(figsize=(12, 6))
    for name in structure_names:
        if name not in all_data:
            continue
        plot_df = all_data[name]
        # Average across targets
        avg_df = plot_df.groupby("num_examples", as_index=False)[["tv_model", "tv_naive", "tv_bayes"]].mean()
        ax.plot(avg_df["num_examples"], avg_df["tv_model"], marker="o", label=f"{name.capitalize()} (Model)", linewidth=2)
        ax.plot(avg_df["num_examples"], avg_df["tv_naive"], marker="s", linestyle="--", alpha=0.6, label=f"{name.capitalize()} (Naive)")
        ax.plot(avg_df["num_examples"], avg_df["tv_bayes"], marker="^", linestyle=":", alpha=0.6, label=f"{name.capitalize()} (Bayes)")
    
    ax.set_xlabel("Number of Examples", fontsize=14)
    ax.set_ylabel("Average TV Distance", fontsize=14)
    ax.set_title("Comparison Across Structures (Averaged Over Targets)", fontsize=16, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    comparison_path = output_path.replace(".png", "_comparison.png")
    plt.savefig(comparison_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Comparison plot saved to: {comparison_path}")



def main():
    print("Getting arguments...")
    args = get_args()

    print("Creating mixed graph structures (5 nodes each)...")
    structures = get_mixed_graph_structures(seed=args.seed)
    structure_names = get_structure_names()
    
    print(f"Structures: {structure_names}")
    
    # Compile templates for all structures
    print("Compiling templates...")
    templates = [compile_template_from_structure(bn) for bn in structures]
    
    # Verify all have 5 nodes
    for i, template in enumerate(templates):
        print(f"  {structure_names[i]}: {template.num_nodes} nodes, "
              f"topo order: {template.topo_nodes}")
        assert template.num_nodes == 5, f"Structure {i} must have 5 nodes"

    pl.seed_everything(args.seed, workers=False)

    print(f"Initializing graph parameters ({args.train_size} per structure)...")
    
    # Initialize parameters for each structure
    p1_lists_train = []
    for i, template in enumerate(templates):
        p1_list = init_graph_params_uniform(
            template, 
            num_graphs=args.train_size, 
            seed=args.seed + i  # Different seed per structure
        )
        p1_lists_train.append(p1_list)
        print(f"  {structure_names[i]}: {args.train_size} graph instances")

    print("Creating batch specification...")
    # Determine if using fixed or dynamic context length
    if args.context_len is not None:
        # Fixed context length
        print(f"Using fixed context length: {args.context_len}")
        spec = MixedICLBatchSpec(
            batch_graphs=args.batch_size,
            target_index=args.target_index,
            num_example=args.context_len,
            device=None,     # keep on CPU; Lightning moves to GPU automatically
            dtype=torch.long,
        )
    else:
        # Dynamic context length (random per batch)
        if args.min_context_len >= args.max_context_len:
            raise ValueError(f"min_context_len ({args.min_context_len}) must be < max_context_len ({args.max_context_len})")
        print(f"Using dynamic context length: {args.min_context_len} to {args.max_context_len}")
        spec = MixedICLBatchSpec(
            batch_graphs=args.batch_size,
            target_index=args.target_index,
            num_example=None,  # Use dynamic sampling
            min_context_len=args.min_context_len,
            max_context_len=args.max_context_len,
            device=None,     # keep on CPU; Lightning moves to GPU automatically
            dtype=torch.long,
        )

    print("Creating training dataset...")
    train_ds = MixedGraphICLSequenceDataset(
        templates=templates,
        p1_lists=p1_lists_train,
        structure_names=structure_names,
        seed=args.seed,
        spec=spec,
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=None,
        num_workers=4,
        pin_memory=True,
    )

    input_dim = 5 + 1  # 5 nodes + 1 target index feature
    
    # Determine max sequence length (context + 1 test token)
    if args.context_len is not None:
        max_seq_len = args.context_len + 1
    else:
        max_seq_len = args.max_context_len + 1
    # Add some buffer for safety
    max_seq_len = max(max_seq_len, 500 + 1)
    
    lit = ICLLightningModule(
        input_dim=input_dim,
        init_lr=args.init_lr,
        weight_decay=1e-2,
        max_steps=args.train_step,
        warmup_steps=args.warmup_steps,
        min_lr=args.min_lr,
        loss_type="l1",  # Use L1 loss (TV distance) to match evaluation metric
        n_embd=256,
        n_layer=12,
        n_head=8,
        dropout=0.1,
        max_seq_len=max_seq_len,
        disable_causal=True,
    )

    # ---- Logging + Trainer
    # Construct run directory path
    if args.context_len is not None:
        context_str = str(args.context_len)
    else:
        context_str = f"{args.min_context_len}to{args.max_context_len}"
    run_name = f"mixed_seed{args.seed}_ctx{context_str}_train{args.train_size}"
    run_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    
    logger = CSVLogger(save_dir=run_dir, name="logs")

    ckpt_cb = ModelCheckpoint(
        monitor="train/loss",
        mode="min",
        save_top_k=1,
        filename="best",
    )

    print("Creating trainer...")
    torch.set_float32_matmul_precision('high')

    trainer = Trainer(
        callbacks=[ckpt_cb],
        max_steps=args.train_step,
        accelerator="auto",
        devices="auto",
        logger=logger,
        log_every_n_steps=100,
        enable_checkpointing=True,
        default_root_dir=run_dir,
        gradient_clip_val=1.0,
    )

    print("Training on mixed structures...")
    print(f"  Batch size: {args.batch_size} (split across {len(structure_names)} structures)")
    if args.context_len is not None:
        print(f"  Context length: {args.context_len} (fixed)")
    else:
        print(f"  Context length: {args.min_context_len}-{args.max_context_len} (dynamic)")
    print(f"  Training steps: {args.train_step}")
    print(f"  Output directory: {run_dir}")
    
    # Resume from checkpoint if specified
    if args.resume_from:
        print(f"\n⚡ Resuming training from checkpoint: {args.resume_from}")
        trainer.fit(lit, train_dataloaders=train_loader, ckpt_path=args.resume_from)
    else:
        trainer.fit(lit, train_dataloaders=train_loader)
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"Model saved to: {run_dir}")
    
    # Load the best checkpoint for evaluation
    best_ckpt_path = ckpt_cb.best_model_path
    print(f"\nLoading best checkpoint for evaluation:")
    print(f"  Path: {best_ckpt_path}")
    print(f"  Best train/loss: {ckpt_cb.best_model_score:.6f}")
    
    lit_best = ICLLightningModule.load_from_checkpoint(best_ckpt_path)
    trained_model = lit_best.model
    trained_model.eval()
    
    # Auto-evaluate on all structures with baselines
    print("\nStarting automatic evaluation...")
    eval_dir = evaluate_on_structures(args, trained_model, templates, structure_names, run_dir)
    
    # Auto-generate plots
    print("\nGenerating visualization...")
    plot_output_path = os.path.join(run_dir, "eval_mixed_results.png")
    plot_mixed_results(eval_dir, structure_names, plot_output_path)
    
    print("\n" + "=" * 70)
    print("ALL DONE!")
    print("=" * 70)
    print(f"Checkpoint: {run_dir}")
    print(f"Evaluation: {eval_dir}")
    print(f"Plots: {plot_output_path}")


if __name__ == "__main__":
    print("Starting mixed structure training...")
    main()
