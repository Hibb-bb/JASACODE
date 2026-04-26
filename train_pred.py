"""
Train the same binary ICL transformer as train.py, but with *prediction* loss:
binary cross-entropy on the test token's observed class (0/1), instead of L1 to
the empirical CPT probability in batch["y"].

Evaluation: TV vs true CPT (evaluate_tv_over_context_with_baselines).

Runs training for a fixed list of seeds (TRAIN_SEEDS), then writes one figure
matching quick_plot.py: 2x2 panels with mean ± std across seeds.

Only graph structures: tree, chain, general (single fixed DAG each).
"""

import argparse
import os
from pathlib import Path
from typing import Any, Callable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from data import (
    ICLBatchSpec,
    MultiGraphICLSequenceDataset,
    compile_template_from_structure,
    init_graph_params_beta,
)
from data.graphs import get_chain, get_general, get_tree
from model import NonCausalGPT2BinaryHead
from utils import EvalSpec, evaluate_tv_over_context_with_baselines

# --- Fixed multi-seed training (edit here) ---
TRAIN_SEEDS: List[int] = [1111, 2222, 3333, 4444, 5555]

ALLOWED_GRAPHS = frozenset({"tree", "chain", "general"})

# Plot style (same spirit as quick_plot.py)
FONT_SIZE = 20
PLOT_RC = {
    "font.size": FONT_SIZE,
    "axes.titlesize": FONT_SIZE,
    "axes.labelsize": FONT_SIZE,
    "xtick.labelsize": FONT_SIZE - 4,
    "ytick.labelsize": FONT_SIZE - 4,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
}

# Figure title by graph (tree/chain/general only)
GRAPH_SUPTITLE = {
    "tree": "Tree (7 Node)",
    "chain": "Chain (7 Node)",
    "general": "General (5 Node)",
}


def get_graph_builder(name: str) -> Callable[[int], object]:
    if name == "tree":
        return get_tree
    if name == "chain":
        return get_chain
    if name == "general":
        return get_general
    raise ValueError(f"graph must be one of {sorted(ALLOWED_GRAPHS)}, got {name!r}")


class ICLBinaryPredLightningModule(pl.LightningModule):
    """
    Binary head; train with BCEWithLogits on hard test labels from batch['full'].
    Logs train/tv = mean |sigmoid(logit) - y_soft| using dataset empirical y (monitoring only).
    """

    def __init__(
        self,
        input_dim: int,
        init_lr: float = 3e-4,
        weight_decay: float = 1e-2,
        max_steps: int = 70000,
        warmup_steps: int = 1000,
        min_lr: float = 0.0,
        **model_kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = NonCausalGPT2BinaryHead(input_dim=input_dim, **model_kwargs)
        self.init_lr = init_lr
        self.weight_decay = weight_decay
        self.max_steps = max_steps
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        x = batch["x"]
        y_soft = batch["y"].float()
        full = batch["full"]
        t = int(batch["target_index"])
        L = int(full.shape[1])
        y_cls = full[:, L - 1, t].float()

        logits = self(x).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logits, y_cls)

        with torch.no_grad():
            p_hat = torch.sigmoid(logits)
            tv_soft = (p_hat - y_soft).abs().mean()

        try:
            lr = self.trainer.optimizers[0].param_groups[0]["lr"]
            self.log("train/lr", lr, prog_bar=False, on_step=True, on_epoch=False)
        except (AttributeError, IndexError, RuntimeError):
            pass

        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/tv", tv_soft, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.init_lr, weight_decay=self.weight_decay)
        warmup = LinearLR(
            opt, start_factor=1e-8, end_factor=1.0, total_iters=self.warmup_steps
        )
        cosine_steps = max(1, self.max_steps - self.warmup_steps)
        cosine = CosineAnnealingLR(opt, T_max=cosine_steps, eta_min=self.min_lr)
        sched = SequentialLR(opt, schedulers=[warmup, cosine], milestones=[self.warmup_steps])
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}


def get_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Binary ICL with BCE; multi-seed train + aggregate plot (tree/chain/general).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument(
        "--context-len",
        type=int,
        default=200,
        help="Fixed number of context examples (ignored if --dynamic-context).",
    )
    p.add_argument(
        "--dynamic-context",
        action="store_true",
        help="Sample context length uniformly in [min-context-len, max-context-len] each batch.",
    )
    p.add_argument("--min-context-len", type=int, default=5)
    p.add_argument("--max-context-len", type=int, default=200)
    p.add_argument(
        "--graph",
        type=str,
        default="tree",
        choices=sorted(ALLOWED_GRAPHS),
        help="Fixed DAG: tree | chain | general only.",
    )
    p.add_argument("--train-size", type=int, default=1000)
    p.add_argument("--test-size", type=int, default=1000)
    p.add_argument(
        "--cpt-beta-alpha",
        type=float,
        default=0.4,
        help="Sample binary CPT entries p ~ Beta(alpha, alpha). alpha>1 => near 0.5, alpha<1 => extreme.",
    )
    p.add_argument("--train-step", type=int, default=50000)
    p.add_argument("--init-lr", type=float, default=3e-4)
    p.add_argument("--warmup-steps", type=int, default=2000)
    p.add_argument("--min-lr", type=float, default=0.0)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--output-dir", type=str, default="outputs_pred")
    p.add_argument(
        "--plot-only",
        action="store_true",
        help="Skip training; only build aggregate plot from existing eval CSVs.",
    )
    return p.parse_args(argv)


def _context_dir_str(args: argparse.Namespace) -> str:
    if getattr(args, "dynamic_context", False):
        return f"{args.min_context_len}to{args.max_context_len}"
    return str(args.context_len)


def _run_dir(args: argparse.Namespace, seed: int) -> str:
    context_str = _context_dir_str(args)
    return os.path.join(
        args.output_dir,
        args.graph,
        f"seed_{seed}",
        context_str,
        f"L{args.num_layers}",
        str(args.train_size),
        f"beta{args.cpt_beta_alpha:g}",
        "pred_bce",
    )


def _evaluate_tv(
    args: argparse.Namespace,
    model: torch.nn.Module,
    run_dir: str,
    seed: int,
) -> None:
    bn = get_graph_builder(args.graph)(seed=seed)
    template = compile_template_from_structure(bn)
    param_rng = np.random.default_rng(seed + 947)
    p1_list = init_graph_params_beta(
        template, num_graphs=args.test_size, alpha=float(args.cpt_beta_alpha), seed=param_rng
    )
    spec = EvalSpec(
        context_lens=[1, 2, 5, 10, 20, 50, 100, 200, 300, 400, 500],
        num_episodes=args.test_size,
        seed=seed,
        output_csv=run_dir + "_eval_tv.csv",
        device="cuda",
        infer_batch_size=16,
    )
    evaluate_tv_over_context_with_baselines(model, template, p1_list, spec)


def _train_one_seed(args: argparse.Namespace, seed: int) -> str:
    """Train + eval for one seed. Returns run_dir."""
    if args.dynamic_context:
        if args.min_context_len >= args.max_context_len:
            raise ValueError("min_context_len must be < max_context_len for dynamic context")
        print(f"Dynamic context: {args.min_context_len}..{args.max_context_len}")
    else:
        if args.context_len < 1:
            raise ValueError("context_len must be >= 1 when not using --dynamic-context")
        print(f"Fixed context length: {args.context_len}")

    print(f"Seed={seed} | graph={args.graph} (BCE); eval = TV vs p_true")

    bn = get_graph_builder(args.graph)(seed=seed)
    template = compile_template_from_structure(bn)
    pl.seed_everything(seed, workers=False)

    p1_list_train = init_graph_params_beta(
        template, num_graphs=args.train_size, alpha=float(args.cpt_beta_alpha), seed=seed
    )

    if not args.dynamic_context:
        spec = ICLBatchSpec(
            batch_graphs=args.batch_size,
            target_index=None,
            num_example=int(args.context_len),
            device=None,
            dtype=torch.long,
        )
    else:
        spec = ICLBatchSpec(
            batch_graphs=args.batch_size,
            target_index=None,
            num_example=None,
            min_context_len=args.min_context_len,
            max_context_len=args.max_context_len,
            device=None,
            dtype=torch.long,
        )

    train_ds = MultiGraphICLSequenceDataset(
        template=template,
        p1_list=p1_list_train,
        seed=seed,
        spec=spec,
        return_full=True,
    )
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=None,
        num_workers=4,
        pin_memory=True,
    )

    input_dim = template.num_nodes + 1
    if not args.dynamic_context:
        max_seq_len = max(int(args.context_len) + 1, 500 + 1)
    else:
        max_seq_len = max(args.max_context_len + 1, 500 + 1)

    lit = ICLBinaryPredLightningModule(
        input_dim=input_dim,
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

    run_dir = _run_dir(args, seed)
    os.makedirs(run_dir, exist_ok=True)
    logger = CSVLogger(save_dir=run_dir, name="logs")
    ckpt_cb = ModelCheckpoint(
        monitor="train/loss",
        mode="min",
        save_top_k=1,
        filename="best",
        save_last=False,
    )

    torch.set_float32_matmul_precision("high")
    trainer = Trainer(
        callbacks=[ckpt_cb],
        max_steps=args.train_step,
        accelerator="auto",
        devices="auto",
        strategy="ddp_find_unused_parameters_true",
        logger=logger,
        log_every_n_steps=100,
        enable_checkpointing=True,
        default_root_dir=run_dir,
        gradient_clip_val=1.0,
    )

    trainer.fit(lit, train_dataloaders=train_loader)

    if trainer.global_rank == 0:
        best = ckpt_cb.best_model_path
        if best and os.path.exists(best):
            trained_model = ICLBinaryPredLightningModule.load_from_checkpoint(
                best, strict=False
            ).model
        else:
            trained_model = lit.model
        trained_model.eval()
        print(f"Evaluating seed={seed} (TV vs true CPT)...")
        _evaluate_tv(args, trained_model, run_dir, seed)

    return run_dir


def _flatten_col(c: Any) -> str:
    if isinstance(c, tuple):
        return f"{c[0]}_{c[1]}" if c[1] else str(c[0])
    return str(c)


def save_aggregate_eval_plot(
    output_dir: str,
    graph: str,
    context_str: str,
    num_layers: int,
    train_size: int,
    cpt_beta_alpha: float,
    seeds: List[int],
    out_path: Path,
) -> None:
    """
    Same layout as quick_plot.py: 2x2, mean ± std across seeds over episodes
    (groupby context_len, target_index then mean; then across seeds).
    """
    rows = []
    for seed in seeds:
        run_dir = os.path.join(
            output_dir,
            graph,
            f"seed_{seed}",
            context_str,
            f"L{num_layers}",
            str(train_size),
            f"beta{cpt_beta_alpha:g}",
            "pred_bce",
        )
        file_path = Path(run_dir + "_eval_tv.csv")
        if not file_path.is_file():
            print(f"WARNING: {file_path} not found, skipping seed {seed}")
            continue
        df = pd.read_csv(file_path)
        seed_vals = (
            df.groupby(["context_len", "target_index"], as_index=False)[
                ["tv_model", "tv_naive", "tv_bayes"]
            ]
            .mean()
            .rename(columns={"context_len": "num_examples"})
        )
        seed_vals["seed"] = seed
        rows.append(seed_vals)

    if not rows:
        raise FileNotFoundError(
            f"No eval CSVs found under {output_dir}/{graph}/.../pred_bce for seeds={seeds}"
        )

    wide_df = pd.concat(rows, ignore_index=True)

    agg = (
        wide_df.groupby(["num_examples", "target_index"])[["tv_model", "tv_naive", "tv_bayes"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    agg.columns = [_flatten_col(c) for c in agg.columns]

    avg_per_seed = (
        wide_df.groupby(["num_examples", "seed"], as_index=False)[["tv_model", "tv_naive", "tv_bayes"]]
        .mean()
    )
    avg_agg = (
        avg_per_seed.groupby("num_examples")[["tv_model", "tv_naive", "tv_bayes"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    avg_agg.columns = [_flatten_col(c) for c in avg_agg.columns]

    plt.rcParams.update(PLOT_RC)
    target_indices = sorted(agg["target_index"].unique())
    n_targets = len(target_indices)
    target_colors = plt.cm.tab10(np.arange(n_targets))

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 9), sharex=True, sharey=True)

    def plot_panel(ax, agg_df, value_mean_col, value_std_col, title, show_legend=True):
        for i, target_idx in enumerate(target_indices):
            sub = agg_df[agg_df["target_index"] == target_idx].sort_values("num_examples")
            x = sub["num_examples"].values
            y = sub[value_mean_col].values
            yerr = sub[value_std_col].values
            if np.any(np.isnan(yerr)):
                yerr = None
            ax.errorbar(
                x,
                y,
                yerr=yerr,
                marker="o",
                label=f"Node {target_idx}" if show_legend else None,
                color=target_colors[i],
                alpha=0.7,
                capsize=3,
                capthick=1,
            )
        ax.set_title(title)
        ax.set_xlabel("Number of Examples")
        ax.set_ylabel("TV Distance")
        ax.grid(True, alpha=0.3)
        if show_legend:
            ax.legend(fontsize=14, loc="upper right")

    plot_panel(axes[0, 0], agg, "tv_model_mean", "tv_model_std", "Transformer", show_legend=True)
    axes[0, 0].set_xlabel("")
    plot_panel(axes[0, 1], agg, "tv_bayes_mean", "tv_bayes_std", "Bayesian Inference", show_legend=False)
    axes[0, 1].set_xlabel("")
    axes[0, 1].set_ylabel("")
    plot_panel(axes[1, 0], agg, "tv_naive_mean", "tv_naive_std", "Naive Inference", show_legend=False)

    ax_tr = axes[1, 1]
    for col, label in [
        ("tv_model", "Transformer"),
        ("tv_naive", "Naive"),
        ("tv_bayes", "Bayes"),
    ]:
        sub = avg_agg.sort_values("num_examples")
        x = sub["num_examples"].values
        y = sub[f"{col}_mean"].values
        yerr = sub[f"{col}_std"].values
        if np.any(np.isnan(yerr)):
            yerr = None
        ax_tr.errorbar(x, y, yerr=yerr, marker="o", label=label, capsize=3, capthick=1)
    ax_tr.set_title("Averaged Across Nodes")
    ax_tr.set_xlabel("Number of Examples")
    ax_tr.set_ylabel("")
    ax_tr.grid(True, alpha=0.3)
    ax_tr.legend(fontsize=14, loc="upper right")
    axes[1, 1].set_ylabel("")

    for ax, label in zip(axes.flat, ["(a)", "(b)", "(c)", "(d)"]):
        ax.text(
            0.02,
            0.98,
            label,
            transform=ax.transAxes,
            fontsize=FONT_SIZE,
            fontweight="bold",
            va="top",
            ha="left",
        )

    title_str = GRAPH_SUPTITLE.get(graph, graph)
    fig.suptitle(title_str, fontsize=FONT_SIZE + 6, y=0.90)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close()
    print(f"Saved aggregate plot: {out_path}")


def main() -> None:
    args = get_args()

    if not args.dynamic_context and args.context_len < 1:
        raise ValueError("--context-len must be >= 1 unless --dynamic-context is set")

    context_str = _context_dir_str(args)

    repo_root = Path(__file__).resolve().parent
    fig_out = repo_root / "imgs" / f"{args.graph}_pred_avg{len(TRAIN_SEEDS)}.png"

    if not args.plot_only:
        for seed in TRAIN_SEEDS:
            _train_one_seed(args, seed)

    save_aggregate_eval_plot(
        output_dir=args.output_dir,
        graph=args.graph,
        context_str=context_str,
        num_layers=args.num_layers,
        train_size=args.train_size,
        cpt_beta_alpha=float(args.cpt_beta_alpha),
        seeds=list(TRAIN_SEEDS),
        out_path=fig_out,
    )


if __name__ == "__main__":
    main()
