# JASACODE — short runbook

## Setup

From the repo root:

```bash
uv sync
```

That installs everything in `pyproject.toml` (Python 3.12+).

Either prefix commands with `uv run`, or activate the venv:

```bash
source .venv/bin/activate
```

---

## Experiments (three shell drivers)

### 1) Single structure — table-style training

**Script:** `single-structure.sh`  
**Runs:** `train.py` for several graphs and five seeds, then `quick_plot.py` per graph.  
**Outputs:** under `runs/` (exact folders are built inside `train.py`).

### 2) Single structure — next-token training

**Script:** `pred.sh`  
**Runs:** `train_pred.py` for `tree`, `chain`, and `general`.  
**Outputs:** under `outputs_pred/…/pred_bce/` (exact folders are built inside `train_pred.py`).

### 3) Sachs — score a saved model on real discretized data

**Script:** `sachs_l12.sh`  
**Runs:** `eval_sachs_real.py`. The commented block shows optional `train_sachs.py` plus eval if you turn it back on.  
**Needs:** a trained checkpoint path and `Sachs/disc_data/*.csv`.

---

## Plotting

| File | Use |
|------|-----|
| `quick_plot.py` | Mean ± std over five seeds from eval CSVs → figure under `imgs/`. |
| `quick_plot_sachs.py` | Same style for real-Sachs eval outputs (e.g. under `runs/sachs_real_eval/…`). |
| `plot_loss.py` | Loss and TV from Lightning `metrics.csv`. |
| `plot_loss_runs.sh` | Calls `plot_loss.py` for tree / chain / general paths aligned with `pred.sh` defaults. |

Edit paths or flags in the scripts if your run folders differ.

---

## Notes

- The `.sh` files may include Slurm lines; change account, partition, time, and email for your cluster.
- If a run folder is missing, plot helpers may warn or skip—run the matching train job first.
