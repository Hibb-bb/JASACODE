import numpy as np
import pandas as pd
from pathlib import Path

def _mutual_information(x: np.ndarray, y: np.ndarray) -> float:
    """
    Mutual information I(X;Y) for discrete integer arrays x,y.
    Uses natural log. Returns >= 0.
    """
    x = x.astype(np.int64)
    y = y.astype(np.int64)
    n = x.size
    if n == 0:
        return 0.0

    # Map to contiguous labels to make contingency compact
    x_vals, x_inv = np.unique(x, return_inverse=True)
    y_vals, y_inv = np.unique(y, return_inverse=True)

    kx = x_vals.size
    ky = y_vals.size
    # contingency
    idx = x_inv * ky + y_inv
    counts = np.bincount(idx, minlength=kx * ky).reshape(kx, ky).astype(np.float64)

    pxy = counts / n
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)

    # Avoid log(0): only evaluate where pxy > 0
    nz = pxy > 0
    mi = (pxy[nz] * (np.log(pxy[nz]) - np.log(px[nz.any(axis=1), :][..., 0][:, None][nz] / 1.0)  # not used
                     ) )

    # The above got messy; do it cleanly:
    # Compute log(pxy/(px*py)) only on nz using broadcasting
    # Recompute px, py flattened for indexing
    px_flat = pxy.sum(axis=1)
    py_flat = pxy.sum(axis=0)

    mi = 0.0
    for i in range(kx):
        for j in range(ky):
            p = pxy[i, j]
            if p > 0:
                mi += p * (np.log(p) - np.log(px_flat[i]) - np.log(py_flat[j]))
    return float(mi)


def _mutual_information_fast(x: np.ndarray, y: np.ndarray) -> float:
    """
    Same as _mutual_information, but vectorized and clearer.
    """
    x = x.astype(np.int64)
    y = y.astype(np.int64)
    n = x.size
    if n == 0:
        return 0.0

    x_vals, x_inv = np.unique(x, return_inverse=True)
    y_vals, y_inv = np.unique(y, return_inverse=True)
    kx = x_vals.size
    ky = y_vals.size

    idx = x_inv * ky + y_inv
    counts = np.bincount(idx, minlength=kx * ky).reshape(kx, ky).astype(np.float64)
    pxy = counts / n
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)

    nz = pxy > 0
    mi = (pxy[nz] * (np.log(pxy[nz]) - np.log(px[nz.any(axis=1)])[:, None].repeat(ky, axis=1)[nz]  # fallback
                     ) )
    # Above line is not robust; use clean broadcasting:
    mi = float(np.sum(pxy[nz] * (np.log(pxy[nz]) - np.log(px.repeat(ky, axis=1)[nz]) - np.log(py.repeat(kx, axis=0)[nz]))))
    return mi


def _initial_discretize_quantile(col: pd.Series, ibreaks: int) -> np.ndarray:
    """
    Initial marginal discretization into ibreaks bins using empirical quantiles (qcut).
    Returns integer bins 0..k-1 (k may be < ibreaks if duplicates/ties).
    """
    # pandas.qcut can drop bins if many ties. That's fine; we keep what we can.
    binned = pd.qcut(col, q=ibreaks, labels=False, duplicates="drop")
    if binned.isna().any():
        # If qcut fails due to constant column, put everything in one bin.
        binned = binned.fillna(0)
    return binned.astype(int).to_numpy()


def discretize_hartemink_bnlearn(
    df: pd.DataFrame,
    breaks: int = 3,
    ibreaks: int = 60,
    idisc: str = "quantile",
    return_labels: bool = True,
) -> pd.DataFrame:
    """
    Hartemink information-preserving discretization following bnlearn's call:
      discretize(method="hartemink", breaks=3, ibreaks=60, idisc="quantile")

    Steps:
      1) initial marginal discretization for each variable into ibreaks bins via quantiles
      2) iteratively merge adjacent bins to reach 'breaks' bins per variable,
         choosing merges that minimize loss of pairwise mutual information with other variables.

    Output:
      - if return_labels=True: categorical labels {"LOW","AVG","HIGH"} (or generalised levels)
      - else: integer codes 0..(breaks-1)
    """
    if idisc.lower() != "quantile":
        raise ValueError("This implementation matches bnlearn Sachs setup: idisc='quantile' only.")

    X = df.copy()
    cols = list(X.columns)
    n_vars = len(cols)

    # 1) initial discretization
    disc = {c: _initial_discretize_quantile(X[c], ibreaks) for c in cols}

    # Ensure bins are contiguous 0..k-1
    for c in cols:
        _, inv = np.unique(disc[c], return_inverse=True)
        disc[c] = inv.astype(np.int64)

    # Precompute MI between every pair for current discretization
    def total_mi_for_var(var: str, candidate_x: np.ndarray, disc_current: dict) -> float:
        s = 0.0
        for other in cols:
            if other == var:
                continue
            s += _mutual_information_fast(candidate_x, disc_current[other])
        return s

    # Helper: merge adjacent levels a and a+1 in array x
    def merge_adjacent(x: np.ndarray, a: int) -> np.ndarray:
        # Merge level a+1 into a, then relabel to keep contiguous
        x2 = x.copy()
        x2[x2 == a + 1] = a
        # Shift down levels > a+1
        x2[x2 > a + 1] -= 1
        return x2

    # 2) iterative merging until each variable has <= breaks levels
    # Greedy global-best merge across all variables/adjacent-pairs (common Hartemink variant).
    while True:
        levels = {c: int(np.max(disc[c]) + 1) for c in cols}
        need = [c for c in cols if levels[c] > breaks]
        if not need:
            break

        best = None  # (loss, var, a, new_x)
        for var in need:
            x = disc[var]
            k = int(np.max(x) + 1)
            # current MI sum for this variable
            mi_current = total_mi_for_var(var, x, disc)

            # try all adjacent merges
            for a in range(k - 1):
                x_new = merge_adjacent(x, a)
                mi_new = total_mi_for_var(var, x_new, disc)
                loss = mi_current - mi_new  # want minimal loss
                if (best is None) or (loss < best[0]):
                    best = (loss, var, a, x_new)

        # apply best merge found
        _, var, _, x_new = best
        disc[var] = x_new

    # Build final DataFrame
    out_int = pd.DataFrame({c: disc[c] for c in cols}, index=df.index)

    print(out_int.head())

    if not return_labels:
        return out_int

    # Map to labels (ordered by increasing bin index): LOW/AVG/HIGH for breaks=3
    if breaks == 3:
        mapping = {0: "LOW", 1: "AVG", 2: "HIGH"}
        return out_int.replace(mapping)

    # Generic labels if breaks != 3
    labels = [f"LEVEL{i}" for i in range(breaks)]
    mapping = {i: labels[i] for i in range(breaks)}
    return out_int.replace(mapping)



def read_sachs_csvs(
    path: str = "/home/dennis/JASACODE/Sachs",
    exclude: str = "GroundTruth.csv",
    return_names: bool = False,
):
    """Read all CSV files in path, except the file named exclude (default GroundTruth.csv).

    Returns:
        If return_names=False: list of DataFrames (as before).
        If return_names=True: list of (filename, DataFrame) tuples.
    """
    folder = Path(path)
    out = []
    for f in sorted(folder.glob("*.csv")):
        if f.name == exclude:
            continue
        df = pd.read_csv(f)
        out.append((f.name, df) if return_names else df)
    return out


def discretize_sachs(
    path: str = "/home/dennis/JASACODE/Sachs",
    breaks: int = 3,
    ibreaks: int = 60,
    return_labels: bool = True,
    save_dir: str | None = None,
):
    """Discretize all Sachs CSVs (except GroundTruth) using Hartemink on the pooled data.

    Learns discretization on the concatenated data, then returns (or saves) one
    discretized DataFrame per original CSV.

    Returns:
        list of (filename, discretized_DataFrame) tuples.

    If save_dir is set, each discretized table is written to save_dir as
    <stem>_disc.csv (e.g. 1.csv -> 1_disc.csv).
    """
    name_df_pairs = read_sachs_csvs(path, return_names=True)
    if not name_df_pairs:
        return []

    names, dfs = zip(*name_df_pairs)
    lengths = [len(df) for df in dfs]
    pooled = pd.concat(dfs, axis=0, ignore_index=True)
    disc_pooled = discretize_hartemink_bnlearn(
        pooled, breaks=breaks, ibreaks=ibreaks, idisc="quantile", return_labels=return_labels
    )

    # Split discretized pooled back into one df per original file
    start = 0
    result = []
    folder = Path(path)
    for name, length in zip(names, lengths):
        end = start + length
        disc_df = disc_pooled.iloc[start:end].reset_index(drop=True)
        result.append((name, disc_df))
        if save_dir:
            out_path = Path(save_dir) / (Path(name).stem + "_disc.csv")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            disc_df.to_csv(out_path, index=False)
        start = end

    return result


# discretize_sachs()