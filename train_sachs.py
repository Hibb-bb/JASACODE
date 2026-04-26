from __future__ import annotations

import argparse
from typing import Optional
import os

import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from data import (
    get_sachs,
    compile_template_from_categorical,
    init_graph_params_categorical,
    ICLBatchSpec,
    MultiGraphICLSequenceDatasetCategorical,
    CategoricalTemplate,
)
from utils import (
    ICLLightningModuleCategorical,
    EvalSpec,
    evaluate_tv_over_context_categorical_with_baselines,
)


def get_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train transformer on synthetic BN ICL data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data / batch
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Number of graphs per batch (B).",
    )
    parser.add_argument(
        "--context-len",
        type=int,
        default=500,
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
        "--graph",
        type=str,
        default="sachs"
    )

    parser.add_argument(
        "--train-size",
        type=int,
        default=10000,
        help="Number of graphs used for training.",
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=1000,
        help="Number of graphs used for testing.",
    )
    parser.add_argument(
        "--cpt-dirichlet-alpha",
        type=float,
        default=0.1,
        help="Symmetric Dirichlet α per class for random training/test CPT rows. "
        "<1 = more peaked (less uniform); 1 = uniform on simplex; >1 = closer to flat 1/K.",
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
        default=2000,
        help="Number of warmup steps for learning rate scheduler.",
    )
    parser.add_argument(
        "--min-lr",
        type=float,
        default=0.0,
        help="Minimum learning rate for cosine decay (default: 0.0).",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=4,
        help="Number of transformer layers.",
    )

    parser.add_argument(
        "--precision",
        type=str,
        default="32-true",
        choices=["32-true", "16-mixed", "bf16-mixed"],
        help="Lightning precision setting. Use bf16-mixed (recommended on A100/H100) or 16-mixed to reduce memory.",
    )
    parser.add_argument(
        "--accumulate-grad-batches",
        type=int,
        default=1,
        help="Accumulate gradients over this many batches to simulate a larger batch size with lower memory.",
    )

    # Output / misc
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory to save checkpoints and logs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    args = parser.parse_args(argv)
    return args


def evaluate(args, model, run_dir):
    """Evaluate on 3-class Sachs: fixed test graphs and categorical TV."""
    bn = get_sachs(seed=args.seed + 999)
    template = compile_template_from_categorical(bn)
    param_rng = np.random.default_rng(args.seed + 1000)
    cpt_list = init_graph_params_categorical(
        template,
        num_graphs=args.test_size,
        seed=param_rng,
        dirichlet_alpha=args.cpt_dirichlet_alpha,
    )
    spec = EvalSpec(
        context_lens=[1, 2, 5, 10, 20, 50, 100, 200, 300, 400, 500],
        num_episodes=args.test_size,
        seed=123,
        output_csv=run_dir + "_eval_tv.csv",
        device="cuda",
        infer_batch_size=32,
    )
    evaluate_tv_over_context_categorical_with_baselines(
        model, template, cpt_list, spec
    )


def plot_results(args, eval_csv_path, output_path):
    """
    Generate plots from evaluation CSV results.
    Creates 4 subplots in a 2x2 grid:
    (1) Top left: Transformer with all target indices
    (2) Top right: Average across all target indices with all 3 baselines
    (3) Bottom left: Naive baseline with all target indices
    (4) Bottom right: Bayesian baseline with all target indices
    Single seed version (no std).
    """
    df = pd.read_csv(eval_csv_path)
    
    # Aggregate per (context_len, target_index) - already averaged across episodes
    plot_df = (
        df.groupby(["context_len", "target_index"], as_index=False)[
            ["tv_model", "tv_naive", "tv_bayes"]
        ]
        .mean()
        .rename(columns={"context_len": "num_examples"})
    )
    
    # Get unique target indices for consistent colors
    target_indices = sorted(plot_df["target_index"].unique())
    n_targets = len(target_indices)
    target_colors = plt.cm.tab10(range(n_targets))
    
    # Create 2x2 subplots
    fig, axes = plt.subplots(
        nrows=2, ncols=2,
        figsize=(14, 10),
        sharex=True, sharey=True
    )
    
    # ===== (1) Top left: Transformer with all target indices =====
    ax_tl = axes[0, 0]
    for target_idx in target_indices:
        target_data = plot_df[plot_df["target_index"] == target_idx].sort_values("num_examples")
        ax_tl.plot(target_data["num_examples"], target_data["tv_model"], 
                  marker="o", label=f"Target {target_idx}", 
                  color=target_colors[target_indices.index(target_idx)], alpha=0.7)
    
    ax_tl.set_title("Transformer - All Target Indices")
    ax_tl.set_xlabel("Number of Examples")
    ax_tl.set_ylabel("TV Distance")
    ax_tl.grid(True, alpha=0.3)
    ax_tl.legend(bbox_to_anchor=(1.05, 1), loc="upper left", frameon=False, fontsize=8)
    
    # ===== (2) Top right: Average across all target indices with all baselines =====
    ax_tr = axes[0, 1]
    # Average across target_index for each method
    avg_df = (
        plot_df.groupby("num_examples", as_index=False)[
            ["tv_model", "tv_naive", "tv_bayes"]
        ]
        .mean()
    )
    
    # Convert to long format for seaborn
    avg_long = avg_df.melt(
        id_vars="num_examples",
        value_vars=["tv_model", "tv_naive", "tv_bayes"],
        var_name="method",
        value_name="tv"
    )
    
    method_map = {
        "tv_model": "Transformer",
        "tv_naive": "Naive",
        "tv_bayes": "Bayes (known DAG)",
    }
    avg_long["method"] = avg_long["method"].map(method_map)
    
    sns.lineplot(
        data=avg_long.sort_values("num_examples"),
        x="num_examples",
        y="tv",
        hue="method",
        marker="o",
        ax=ax_tr
    )
    
    ax_tr.set_title("Averaged Across All Target Indices")
    ax_tr.set_xlabel("Number of Examples")
    ax_tr.set_ylabel("TV Distance")
    ax_tr.grid(True, alpha=0.3)
    ax_tr.legend(bbox_to_anchor=(1.05, 1), loc="upper left", frameon=False)
    
    # ===== (3) Bottom left: Naive baseline with all target indices =====
    ax_bl = axes[1, 0]
    for target_idx in target_indices:
        target_data = plot_df[plot_df["target_index"] == target_idx].sort_values("num_examples")
        ax_bl.plot(target_data["num_examples"], target_data["tv_naive"], 
                  marker="o", label=f"Target {target_idx}", 
                  color=target_colors[target_indices.index(target_idx)], alpha=0.7)
    
    ax_bl.set_title("Naive Baseline - All Target Indices")
    ax_bl.set_xlabel("Number of Examples")
    ax_bl.set_ylabel("TV Distance")
    ax_bl.grid(True, alpha=0.3)
    ax_bl.legend(bbox_to_anchor=(1.05, 1), loc="upper left", frameon=False, fontsize=8)
    
    # ===== (4) Bottom right: Bayesian baseline with all target indices =====
    ax_br = axes[1, 1]
    for target_idx in target_indices:
        target_data = plot_df[plot_df["target_index"] == target_idx].sort_values("num_examples")
        ax_br.plot(target_data["num_examples"], target_data["tv_bayes"], 
                  marker="o", label=f"Target {target_idx}", 
                  color=target_colors[target_indices.index(target_idx)], alpha=0.7)
    
    ax_br.set_title("Bayesian Baseline - All Target Indices")
    ax_br.set_xlabel("Number of Examples")
    ax_br.set_ylabel("TV Distance")
    ax_br.grid(True, alpha=0.3)
    ax_br.legend(bbox_to_anchor=(1.05, 1), loc="upper left", frameon=False, fontsize=8)
    
    fig.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {output_path}")


def main():

    print("Getting arguments...")
    args = get_args()
    
    # Validate context length arguments
    if args.context_len is None:
        if args.min_context_len >= args.max_context_len:
            raise ValueError(f"min_context_len ({args.min_context_len}) must be < max_context_len ({args.max_context_len})")
        if args.min_context_len < 1:
            raise ValueError(f"min_context_len ({args.min_context_len}) must be >= 1")
        print(f"Using dynamic context length sampling: {args.min_context_len} to {args.max_context_len}")
    else:
        if args.context_len < 1:
            raise ValueError(f"context_len ({args.context_len}) must be >= 1")
        print(f"Using fixed context length: {args.context_len}")

    print(f"Using {args.num_layers} transformer layers")


    bn = get_sachs(seed=args.seed)
    print("Compiling template...")
    template = compile_template_from_categorical(bn)

    pl.seed_everything(args.seed, workers=False)

    print(
        "Initializing graph parameters (3-class Sachs), "
        f"Dirichlet α={args.cpt_dirichlet_alpha} per class..."
    )
    cpt_list_train = init_graph_params_categorical(
        template,
        num_graphs=args.train_size,
        seed=args.seed,
        dirichlet_alpha=args.cpt_dirichlet_alpha,
    )
    print("Creating batch specification...")
    print("Using random target index sampling per batch (all target indices will be trained)")

    if args.context_len is not None:
        print(f"Using fixed context length: {args.context_len}")
        spec = ICLBatchSpec(
            batch_graphs=args.batch_size,
            target_index=None,  # Randomly sample target index per batch
            num_example=args.context_len,
            device=None,     # keep on CPU; Lightning moves to GPU automatically
            dtype=torch.long,
        )
    else:
        print(f"Using dynamic context length: {args.min_context_len} to {args.max_context_len}")
        spec = ICLBatchSpec(
            batch_graphs=args.batch_size,
            target_index=None,  # Randomly sample target index per batch
            num_example=None,  # Use dynamic sampling
            min_context_len=args.min_context_len,
            max_context_len=args.max_context_len,
            device=None,     # keep on CPU; Lightning moves to GPU automatically
            dtype=torch.long,
        )

    print("Creating training dataset (categorical)...")
    train_ds = MultiGraphICLSequenceDatasetCategorical(
        template=template,
        cpt_list=cpt_list_train,
        seed=args.seed,
        spec=spec,
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=None,
        num_workers=4,
        pin_memory=True,
    )

    input_dim = template.num_nodes + 1  # D = N + 1 (target index feature)
    # Determine max sequence length (context + 1 test token)
    if args.context_len is not None:
        max_seq_len = args.context_len + 1
    else:
        max_seq_len = args.max_context_len + 1
    # Add some buffer for safety
    max_seq_len = max(max_seq_len, 500 + 1)
    
    lit = ICLLightningModuleCategorical(
        input_dim=input_dim,
        num_classes=template.cardinality,
        init_lr=args.init_lr,
        max_steps=args.train_step,
        warmup_steps=args.warmup_steps,
        min_lr=args.min_lr,
        n_embd=256,
        n_layer=args.num_layers,
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
    run_dir = os.path.join(
        args.output_dir,
        args.graph,
        f"seed_{args.seed}",
        context_str,
        f"L{args.num_layers}",
        str(args.train_size),
    )
    os.makedirs(run_dir, exist_ok=True)
    logger = CSVLogger(save_dir=run_dir, name="logs")

    ckpt_cb = ModelCheckpoint(
        monitor="train/loss",
        mode="min",
        save_top_k=1,
        filename="best",
        save_last=False,  # Don't save last checkpoint, only best
    )

    print("Creating trainer...")

    torch.set_float32_matmul_precision('high')

    # Multi-GPU: use all visible GPUs (respects CUDA_VISIBLE_DEVICES / Slurm).
    # Standard DDP — avoid ddp_find_unused_parameters_true (extra overhead per step).
    trainer = Trainer(
        callbacks=[ckpt_cb],
        max_steps=args.train_step,
        accelerator="auto",
        devices="auto",
        strategy=DDPStrategy(find_unused_parameters=True),
        logger=logger,
        log_every_n_steps=100,
        enable_checkpointing=True,
        default_root_dir=run_dir,
        gradient_clip_val=1.0,
        precision=args.precision,
        accumulate_grad_batches=args.accumulate_grad_batches,
    )

    print("Training...")
    trainer.fit(lit, train_dataloaders=train_loader)
    
    # Only run evaluation/plotting on rank 0 (for multi-GPU training)
    if trainer.global_rank == 0:
        # Load the best checkpoint (lowest training loss)
        best_ckpt_path = ckpt_cb.best_model_path
        if best_ckpt_path and os.path.exists(best_ckpt_path):
            print(f"Loading best checkpoint from: {best_ckpt_path}")
            # Load checkpoint using Lightning's load_from_checkpoint
            # Parameters are loaded from checkpoint hyperparameters, but we can override if needed
            lit_loaded = ICLLightningModuleCategorical.load_from_checkpoint(
                best_ckpt_path,
                strict=False,  # Allow some flexibility if hyperparameters differ slightly
            )
            trained_model = lit_loaded.model
        else:
            print("Warning: No checkpoint found, using final model state")
            trained_model = lit.model  # fallback to final model
        
        trained_model.eval()
        
        print("Evaluating...")
        eval_csv_path = run_dir + "_eval_tv.csv"
        evaluate(args, trained_model, run_dir)
        
        print("Generating plots...")
        plot_output_path = os.path.join(run_dir, "eval_tv_plot.png")
        plot_results(args, eval_csv_path, plot_output_path)


if __name__ == "__main__":
    print("Starting running...")
    main()


    # salloc -p debug_a100 -t 02:00:00 --gres=gpu:1
    # srun --pty bash -l


