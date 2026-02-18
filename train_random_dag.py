"""
Training script for random DAG structures.

Each batch samples a fresh Erdos-Renyi DAG with N nodes and edge probability p.
All B examples in the batch share the DAG structure but have independent CPTs.
The transformer learns to predict P(X_target=1 | parent_config) from context.
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

from data import (
    compile_template_from_structure,
    init_graph_params_uniform,
    sample_random_dag,
    RandomDAGBatchSpec,
    RandomDAGICLDataset,
    # Fixed structures for generalization evaluation
    get_mixed_graph_structures,
    get_mixed_graph_structures_5node,
    get_structure_names,
)
from utils import ICLLightningModule, evaluate_tv_over_context_with_baselines, EvalSpec


def get_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train transformer on random DAG structures",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data / batch
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Number of graphs per batch (B).")
    parser.add_argument("--num-nodes", type=int, default=5,
                        help="Number of nodes per DAG (N).")
    parser.add_argument("--edge-prob", type=float, default=None,
                        help="Fixed edge probability. If None, use --edge-prob-min/max.")
    parser.add_argument("--edge-prob-min", type=float, default=0.1,
                        help="Min edge probability (used when --edge-prob is None).")
    parser.add_argument("--edge-prob-max", type=float, default=0.8,
                        help="Max edge probability (used when --edge-prob is None).")
    parser.add_argument("--context-len", type=int, default=None,
                        help="Fixed context length. If None, use dynamic sampling.")
    parser.add_argument("--min-context-len", type=int, default=5,
                        help="Min context length for dynamic sampling.")
    parser.add_argument("--max-context-len", type=int, default=500,
                        help="Max context length for dynamic sampling.")

    # Training
    parser.add_argument("--train-step", type=int, default=100_000,
                        help="Number of training steps.")
    parser.add_argument("--init-lr", type=float, default=3e-4,
                        help="Initial learning rate.")
    parser.add_argument("--warmup-steps", type=int, default=1000,
                        help="Warmup steps for LR scheduler.")
    parser.add_argument("--min-lr", type=float, default=0.0,
                        help="Minimum learning rate for cosine decay.")

    # Evaluation
    parser.add_argument("--test-size", type=int, default=5000,
                        help="Number of test graph instances per fixed structure.")
    parser.add_argument("--num-eval-dags", type=int, default=20,
                        help="Number of random DAGs to evaluate on.")

    # Output / misc
    parser.add_argument("--output-dir", type=str, default="outputs_random_dag",
                        help="Directory to save checkpoints and logs.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed.")
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Path to checkpoint to resume training from.")

    args = parser.parse_args(argv)
    return args


# ====================================================================== #
#  Evaluation helpers                                                      #
# ====================================================================== #

def evaluate_on_fixed_structures(args, model, run_dir):
    """Evaluate on the fixed tree/chain/general structures to test generalization."""
    print("\n" + "=" * 70)
    print("EVALUATING ON FIXED STRUCTURES (generalization test)")
    print("=" * 70)

    N = args.num_nodes
    if N == 5:
        structures = get_mixed_graph_structures_5node(seed=args.seed + 5000)
    elif N == 7:
        structures = get_mixed_graph_structures(seed=args.seed + 5000)
    else:
        print(f"  Skipping fixed-structure eval (no predefined structures for N={N})")
        return None

    structure_names = get_structure_names()

    eval_dir = os.path.join(run_dir + "_eval", "fixed_structures")
    os.makedirs(eval_dir, exist_ok=True)

    param_rng = np.random.default_rng(args.seed + 2000)

    for bn, name in zip(structures, structure_names):
        template = compile_template_from_structure(bn)
        print(f"\n  {name}: {template.num_nodes} nodes, topo={template.topo_nodes}")

        p1_list = init_graph_params_uniform(
            template, num_graphs=args.test_size,
            seed=int(param_rng.integers(0, 1_000_000)),
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
        evaluate_tv_over_context_with_baselines(model, template, p1_list, eval_spec)
        print(f"  -> {output_csv}")

    return eval_dir


def evaluate_on_random_dags(args, model, run_dir):
    """Evaluate on freshly sampled random DAGs."""
    print("\n" + "=" * 70)
    print("EVALUATING ON RANDOM DAGs")
    print("=" * 70)

    N = args.num_nodes
    eval_dir = os.path.join(run_dir + "_eval", "random_dags")
    os.makedirs(eval_dir, exist_ok=True)

    eval_rng = np.random.default_rng(args.seed + 3000)

    for dag_idx in range(args.num_eval_dags):
        # Sample edge probability from the training range
        if args.edge_prob is not None:
            p_edge = args.edge_prob
        else:
            p_edge = float(eval_rng.uniform(args.edge_prob_min, args.edge_prob_max))

        dag = sample_random_dag(N, p_edge, eval_rng)
        template = compile_template_from_structure(dag)

        num_edges = sum(len(pidx) for pidx in template.parent_idx)
        print(f"\n  DAG {dag_idx}: p_edge={p_edge:.2f}, edges={num_edges}, "
              f"topo={template.topo_nodes}")

        p1_list = init_graph_params_uniform(
            template, num_graphs=args.test_size,
            seed=int(eval_rng.integers(0, 1_000_000)),
        )

        output_csv = os.path.join(eval_dir, f"eval_tv_dag{dag_idx}_p{p_edge:.2f}.csv")
        eval_spec = EvalSpec(
            context_lens=[1, 2, 5, 10, 20, 50, 100, 200, 300, 400, 500],
            num_episodes=args.test_size,
            seed=123,
            output_csv=output_csv,
            device="cuda" if torch.cuda.is_available() else "cpu",
            infer_batch_size=512,
        )
        evaluate_tv_over_context_with_baselines(model, template, p1_list, eval_spec)
        print(f"  -> {output_csv}")

    return eval_dir


# ====================================================================== #
#  Main                                                                    #
# ====================================================================== #

def main():
    print("Getting arguments...")
    args = get_args()

    N = args.num_nodes
    print(f"Random DAG training: N={N} nodes")

    # Determine edge probability description
    if args.edge_prob is not None:
        edge_str = f"p={args.edge_prob}"
        print(f"  Edge probability: {args.edge_prob} (fixed)")
    else:
        edge_str = f"p={args.edge_prob_min}to{args.edge_prob_max}"
        print(f"  Edge probability: Uniform({args.edge_prob_min}, {args.edge_prob_max})")

    pl.seed_everything(args.seed, workers=False)

    # ---- Build batch spec
    print("Creating batch specification...")
    if args.context_len is not None:
        print(f"  Context length: {args.context_len} (fixed)")
        spec = RandomDAGBatchSpec(
            batch_graphs=args.batch_size,
            num_nodes=N,
            edge_prob=args.edge_prob if args.edge_prob is not None else 0.5,
            edge_prob_min=None if args.edge_prob is not None else args.edge_prob_min,
            edge_prob_max=None if args.edge_prob is not None else args.edge_prob_max,
            num_example=args.context_len,
            dtype=torch.long,
            device=None,
        )
    else:
        if args.min_context_len >= args.max_context_len:
            raise ValueError("min_context_len must be < max_context_len")
        print(f"  Context length: {args.min_context_len} to {args.max_context_len} (dynamic)")
        spec = RandomDAGBatchSpec(
            batch_graphs=args.batch_size,
            num_nodes=N,
            edge_prob=args.edge_prob if args.edge_prob is not None else 0.5,
            edge_prob_min=None if args.edge_prob is not None else args.edge_prob_min,
            edge_prob_max=None if args.edge_prob is not None else args.edge_prob_max,
            num_example=None,
            min_context_len=args.min_context_len,
            max_context_len=args.max_context_len,
            dtype=torch.long,
            device=None,
        )

    # ---- Dataset
    print("Creating random DAG dataset...")
    train_ds = RandomDAGICLDataset(seed=args.seed, spec=spec)

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=None,
        num_workers=4,
        pin_memory=True,
    )

    # ---- Model
    input_dim = N + 1  # N nodes + 1 target index feature

    if args.context_len is not None:
        max_seq_len = args.context_len + 1
    else:
        max_seq_len = args.max_context_len + 1
    max_seq_len = max(max_seq_len, 501)

    lit = ICLLightningModule(
        input_dim=input_dim,
        init_lr=args.init_lr,
        weight_decay=1e-2,
        max_steps=args.train_step,
        warmup_steps=args.warmup_steps,
        min_lr=args.min_lr,
        loss_type="l1",
        n_embd=256,
        n_layer=12,
        n_head=8,
        dropout=0.1,
        max_seq_len=max_seq_len,
        disable_causal=True,
    )

    # ---- Logging + Trainer
    if args.context_len is not None:
        context_str = str(args.context_len)
    else:
        context_str = f"{args.min_context_len}to{args.max_context_len}"

    run_name = f"rdag_seed{args.seed}_{N}nodes_{edge_str}_ctx{context_str}"
    run_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    logger = CSVLogger(save_dir=run_dir, name="logs")
    ckpt_cb = ModelCheckpoint(
        monitor="train/loss", mode="min", save_top_k=1, filename="best",
    )

    print("Creating trainer...")
    torch.set_float32_matmul_precision("high")

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

    print("=" * 70)
    print("TRAINING")
    print("=" * 70)
    print(f"  Batch size:      {args.batch_size}")
    print(f"  Nodes:           {N}")
    print(f"  Edge prob:       {edge_str}")
    print(f"  Context:         {context_str}")
    print(f"  Training steps:  {args.train_step}")
    print(f"  Output:          {run_dir}")

    if args.resume_from:
        print(f"\n  Resuming from checkpoint: {args.resume_from}")
        trainer.fit(lit, train_dataloaders=train_loader, ckpt_path=args.resume_from)
    else:
        trainer.fit(lit, train_dataloaders=train_loader)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)

    # ---- Load best checkpoint
    best_ckpt_path = ckpt_cb.best_model_path
    print(f"  Best checkpoint: {best_ckpt_path}")
    print(f"  Best train/loss: {ckpt_cb.best_model_score:.6f}")

    lit_best = ICLLightningModule.load_from_checkpoint(best_ckpt_path)
    trained_model = lit_best.model
    trained_model.eval()

    # ---- Evaluation
    # 1. Evaluate on random DAGs
    evaluate_on_random_dags(args, trained_model, run_dir)

    # 2. Evaluate on fixed structures (generalization)
    evaluate_on_fixed_structures(args, trained_model, run_dir)

    print("\n" + "=" * 70)
    print("ALL DONE!")
    print("=" * 70)
    print(f"  Run directory:  {run_dir}")
    print(f"  Eval directory: {run_dir}_eval/")


if __name__ == "__main__":
    print("Starting random DAG training...")
    main()
