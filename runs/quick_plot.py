import pandas as pd
import argparse



parser = argparse.ArgumentParser(
    description="Plot",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

# Data / batch
parser.add_argument(
    "--file-path",
    type=str,
    default="/home/dennis/JASACODE/runs/tree/seed_1111eval_tv.csv",
    help="CSV file",
)

args = parser.parse_args()


df = pd.read_csv(args.file_path)

# Group by both context_len and target_index, compute mean and std of TV
stats = df.groupby(["context_len", "target_index"])["tv"].agg(["mean", "std"]).reset_index()
stats.columns = ["context_len", "target_index", "tv_mean", "tv_std"]

# Optionally, print for each target_index separately
for target_idx in sorted(df["target_index"].unique()):
    print(f"\n--- Target Index {target_idx} ---")
    target_stats = stats[stats["target_index"] == target_idx]
    print(target_stats[["context_len", "tv_mean", "tv_std"]])
