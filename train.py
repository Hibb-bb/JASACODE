from __future__ import annotations

import argparse
from typing import Optional
import os

import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from data import (
    compile_template_from_structure,
    init_graph_params_beta,
    ICLBatchSpec,
    MultiGraphICLSequenceDataset,
    get_chain,
    get_tree,
    get_general,
)
from utils import evaluate_tv_over_context, ICLLightningModule, EvalSpec


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
        default=100,
        help="Number of context examples per sequence.",
    )
    parser.add_argument(
        "--graph",
        type=str,
        default="tree",
        help="Graph structure name (e.g. tree, chain, collider).",
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=1000,
        help="Number of graphs used for training.",
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=1000,
        help="Number of graphs used for testing.",
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
        default="outputs",
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


def evaluate(args, model, run_dir):

    if args.graph == "tree":
        bn = get_tree()

    elif args.graph == "general":
        bn = get_general()

    elif args.graph == "chain":
        bn = get_chain()

    template = compile_template_from_structure(bn)
    # list of (1, 2^k)
    p1_list = init_graph_params_beta(
        template, num_graphs=1, mode="easy", seed=7777
    )

    # 4) evaluate
    spec = EvalSpec(
        context_lens=[1, 2, 5, 10, 20, 50, 100, 200,  300, 400, 500],
        num_episodes=args.test_size,
        seed=123,
        output_csv=run_dir + "eval_tv.csv",
        device="cuda",
        infer_batch_size=512,
    )
    evaluate_tv_over_context(model, template, p1_list, spec)


def main():

    print("Getting arguments...")
    args = get_args()

    if args.graph == "tree":
        target_index = 6
        bn = get_tree()

    elif args.graph == "general":
        target_index = 4
        bn = get_general()

    elif args.graph == "chain":
        target_index = 6
        bn = get_chain()

    print("Compiling template...")

    template = compile_template_from_structure(bn)

    pl.seed_everything(args.seed, workers=False)

    print("Initializing graph parameters...")

    p1_list_train = init_graph_params_beta(
        template, num_graphs=args.train_size, mode="easy", seed=args.seed
    )
    # p1_list_test = init_graph_params_beta(
    #     template, num_graphs=args.test_size, mode="easy", seed=args.seed + 1
    # )
    print("Creating batch specification...")
    # L = context_len + 1
    spec = ICLBatchSpec(
        batch_graphs=args.batch_size,
        num_example=args.context_len,
        # you can make this an arg, or randomize per batch later
        target_index=target_index,
        device=None,     # keep on CPU; Lightning moves to GPU automatically
        dtype=torch.long,
    )

    print("Creating training dataset...")
    train_ds = MultiGraphICLSequenceDataset(
        template=template,
        p1_list=p1_list_train,
        seed=args.seed,
        spec=spec,
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=None,
        num_workers=255,
        pin_memory=True,
    )

    input_dim = template.num_nodes + 1  # D = N + 1 (target index feature)
    lit = ICLLightningModule(
        input_dim=input_dim,
        init_lr=args.init_lr,
        n_embd=256,
        n_layer=6,
        n_head=8,
        dropout=0.1,
        max_seq_len=args.context_len + 1,
        disable_causal=True,   # best-effort patch
    )

    # ---- Logging + Trainer
    run_dir = os.path.join(args.output_dir, args.graph, f"seed_{args.seed}")
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
        # precision="32-true",
    )

    print("Training...")
    trainer.fit(lit, train_dataloaders=train_loader)
    trained_model = lit.model  # this is your NonCausalGPT2BinaryHead
    trained_model.eval()
    evaluate(args, trained_model, run_dir)


if __name__ == "__main__":
    print("Starting running...")
    main()
