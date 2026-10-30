"""Generate SLURM array config files (one command per line) for the rebuttal experiments.

Experiments (3 graphs x 5 seeds each):
  ref : gpt2 L4H8 D256, TV loss, Table-2 settings  (reference, truly non-causal attention)
  A1  : gpt2 L2H1 D256, TV loss
  A2  : theory L2H1 D(2N+1)d, TV loss        (+ width sweep D64, D256)
  B   : gpt2 L4H8, BCE prediction loss, Table-3 settings (ctx 500, bs 128, 50k steps)
  C1  : gpt2 L2H1 D256, BCE
  C2  : theory L2H1 D(2N+1)d, BCE            (+ D256)
"""
from pathlib import Path

GRAPHS = ["tree", "chain", "general"]
SEEDS = [1111, 2222, 3333, 4444, 5555]
OUT_TV = "runs_rebuttal/tv"
OUT_BCE = "runs_rebuttal/bce"

TV_COMMON = (
    "python train.py --batch-size 64 --context-len 200 --train-step 100000 "
    "--train-size 20000 --test-size 5000 --warmup-steps 2000 --min-lr 1e-6 --init-lr 3e-4 "
    f"--output-dir {OUT_TV}"
)
BCE_COMMON = (
    "python train_pred.py --batch-size 128 --context-len 500 --train-step 50000 "
    "--train-size 20000 --test-size 5000 --warmup-steps 2000 --min-lr 1e-6 --init-lr 3e-4 "
    f"--cpt-prior uniform --no-plot --output-dir {OUT_BCE}"
)

CONFIGS = {
    # name: (common, model flags)
    "ref": (TV_COMMON, "--num-layers 4 --arch gpt2"),
    "A1": (TV_COMMON, "--num-layers 2 --arch gpt2 --n-head 1"),
    "A2": (TV_COMMON, "--num-layers 2 --arch theory"),
    "A2w64": (TV_COMMON, "--num-layers 2 --arch theory --n-embd 64"),
    "A2w256": (TV_COMMON, "--num-layers 2 --arch theory --n-embd 256"),
    "B": (BCE_COMMON, "--num-layers 4 --arch gpt2"),
    "C1": (BCE_COMMON, "--num-layers 2 --arch gpt2 --n-head 1"),
    "C2": (BCE_COMMON, "--num-layers 2 --arch theory"),
    "C2w256": (BCE_COMMON, "--num-layers 2 --arch theory --n-embd 256"),
}

out_dir = Path("slurm/configs"); out_dir.mkdir(parents=True, exist_ok=True)
for name, (common, flags) in CONFIGS.items():
    lines = []
    for g in GRAPHS:
        for s in SEEDS:
            seed_flag = f"--seed {s}" if "train.py" in common else f"--seeds {s}"
            lines.append(f"{common} --graph {g} {seed_flag} {flags}")
    (out_dir / f"{name}.txt").write_text("\n".join(lines) + "\n")
    print(f"{name}: {len(lines)} runs -> slurm/configs/{name}.txt")
