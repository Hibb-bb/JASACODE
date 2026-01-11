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

from data import (
    compile_template_from_structure,
    init_graph_params_uniform,
    get_mixed_graph_structures,
    get_structure_names,
    MixedICLBatchSpec,
    MixedGraphICLSequenceDataset,
)
from utils import ICLLightningModule


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
        default=10,
        help="Number of context examples per sequence.",
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

    args = parser.parse_args(argv)
    return args


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
    spec = MixedICLBatchSpec(
        batch_graphs=args.batch_size,
        num_example=args.context_len,
        target_index=args.target_index,
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
    lit = ICLLightningModule(
        input_dim=input_dim,
        init_lr=args.init_lr,
        n_embd=256,
        n_layer=12,
        n_head=8,
        dropout=0.1,
        max_seq_len=args.context_len + 1,
        disable_causal=True,
    )

    # ---- Logging + Trainer
    run_name = f"mixed_seed{args.seed}_ctx{args.context_len}_train{args.train_size}"
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
        log_every_n_steps=1000,
        enable_checkpointing=True,
        default_root_dir=run_dir,
        gradient_clip_val=1.0,
    )

    print("Training on mixed structures...")
    print(f"  Batch size: {args.batch_size} (split across {len(structure_names)} structures)")
    print(f"  Context length: {args.context_len}")
    print(f"  Training steps: {args.train_step}")
    print(f"  Output directory: {run_dir}")
    
    trainer.fit(lit, train_dataloaders=train_loader)
    
    print("Training complete!")
    print(f"Model saved to: {run_dir}")


if __name__ == "__main__":
    print("Starting mixed structure training...")
    main()
