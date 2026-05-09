"""
K-fold cross-validation entry point for real-data UDE experiments.

Usage
-----
Supervisor (runs all folds × seeds):
    python train_ude_real_data_kfold.py config.json

Worker (runs one fold × seed, called by supervisor):
    python train_ude_real_data_kfold.py config.json <fold_idx> <seed> <batch_dir>

Config must contain "n_folds" and "cv_seed" in addition to all existing keys.
Symbolic regression and Hill regression are NOT run in either phase.
"""

import jax
import json
import sys
import subprocess
import matplotlib
import jax.numpy as jnp
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

jax.config.update("jax_enable_x64", True)

project_root = Path(__file__).resolve().parent

config_file_path = sys.argv[1]
with open(config_file_path, "r") as f:
    config = json.load(f)

if "n_folds" not in config or "cv_seed" not in config:
    raise ValueError(
        "Config must contain 'n_folds' and 'cv_seed' for k-fold CV. "
        "Add them to the config file before running this script."
    )

# ---------------------------------------------------------------------------
# SUPERVISOR MODE  (python train_ude_real_data_kfold.py config.json)
# ---------------------------------------------------------------------------
if len(sys.argv) == 2:
    from utils.cv_utils import make_kfold_splits

    seeds = config["seed"]
    n_folds = config["n_folds"]
    cv_seed = config["cv_seed"]
    model_name = config["model_name"]

    batch_dir_name = (
        f"ude_{model_name}_kfold_{n_folds}fold_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}"
    )
    batch_dir = project_root / "experiments" / batch_dir_name
    batch_dir.mkdir(parents=True, exist_ok=True)

    with open(batch_dir / "config.json", "w") as f:
        json.dump(config, f, indent=4)

    # Load data only to get n_cells (no JAX needed in supervisor)
    nfkb_df = pd.read_csv(
        project_root / config["nfkb_signal_path"], header=None, index_col=False
    )
    n_cells = nfkb_df.shape[1]

    # Compute splits and save fold_splits.csv for reproducibility
    splits = make_kfold_splits(n_cells, n_folds, cv_seed)
    fold_rows = []
    for fold_id, (_, test_idx) in enumerate(splits):
        for ci in test_idx:
            fold_rows.append({"cell_index": int(ci), "fold_id": fold_id})
    pd.DataFrame(fold_rows).sort_values("cell_index").reset_index(drop=True).to_csv(
        batch_dir / "fold_splits.csv", index=False
    )

    print(f"K-fold CV: {n_folds} folds, seeds={seeds}")
    print(f"Total workers: {n_folds} folds × {len(seeds)} seeds = {n_folds * len(seeds)}")
    print(f"Experiment folder: {batch_dir}")

    for fold_idx in range(n_folds):
        print(f"\n--- Fold {fold_idx} / {n_folds - 1}: spawning {len(seeds)} workers ---")
        train_n = len(splits[fold_idx][0])
        test_n = len(splits[fold_idx][1])
        print(f"    train cells: {train_n}, test cells: {test_n}")

        processes = []
        log_files = []

        for s in seeds:
            worker_dir = batch_dir / f"fold_{fold_idx}" / f"seed_{s}"
            worker_dir.mkdir(parents=True, exist_ok=True)
            log_path = worker_dir / "process_output.log"
            f_log = open(log_path, "w")
            log_files.append(f_log)

            print(f"    Launching fold={fold_idx} seed={s} → {log_path}")
            p = subprocess.Popen(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    config_file_path,
                    str(fold_idx),
                    str(s),
                    str(batch_dir),
                ],
                stdout=f_log,
                stderr=subprocess.STDOUT,
            )
            processes.append(p)

        for p in processes:
            p.wait()
        for f in log_files:
            f.close()

        print(f"Fold {fold_idx} complete.")

    print(f"\nAll {n_folds} folds complete. Results in: {batch_dir}")
    sys.exit(0)

# ---------------------------------------------------------------------------
# WORKER MODE  (called by supervisor with fold_idx, seed, batch_dir)
# ---------------------------------------------------------------------------
elif len(sys.argv) == 5:
    fold_idx = int(sys.argv[2])
    seed = int(sys.argv[3])
    batch_dir = Path(sys.argv[4])
    config["seed"] = seed
    matplotlib.use("Agg")

else:
    raise ValueError(
        f"Unexpected number of arguments ({len(sys.argv)}). "
        "Usage: train_ude_real_data_kfold.py config.json [fold_idx seed batch_dir]"
    )

# ---------------------------------------------------------------------------
# WORKER: run train phase then test phase for this (fold_idx, seed)
# ---------------------------------------------------------------------------
from utils.cv_utils import make_kfold_splits
from utils.real_data_runner import run_one_phase
from utils.models import ude_models

print(f"Worker start: fold={fold_idx}, seed={seed}")

n_folds = config["n_folds"]
cv_seed = config["cv_seed"]
model_name = config["model_name"]

t0 = config["time_start"]
t1 = config["time_end"]
dt = config["time_step"]
time_points = jnp.arange(t0, t1 + dt, dt)
ude_model = ude_models[model_name]

np.random.seed(seed)
master_key = jax.random.PRNGKey(seed)

# Load full dataset
nfkb_signal = jnp.array(
    pd.read_csv(
        project_root / config["nfkb_signal_path"], header=None, index_col=False
    ).values
)
true_p53_values = jnp.array(
    pd.read_csv(
        project_root / config["true_p53_values_path"], header=None, index_col=False
    ).values
)
n_cells = true_p53_values.shape[1]

splits = make_kfold_splits(n_cells, n_folds, cv_seed)
train_idx, test_idx = splits[fold_idx]

fold_seed_dir = batch_dir / f"fold_{fold_idx}" / f"seed_{seed}"
train_dir = fold_seed_dir / "train"
test_dir = fold_seed_dir / "test"

# ---- TRAIN PHASE ----
print(f"Train phase: {len(train_idx)} cells")
master_key, train_phase_key = jax.random.split(master_key)

trained_vars, train_preds, trained_nn = run_one_phase(
    phase_dir=train_dir,
    config=config,
    project_root=project_root,
    cell_indices=train_idx,
    full_nfkb=nfkb_signal,
    full_p53=true_p53_values,
    time_points=time_points,
    ude_model=ude_model,
    key=train_phase_key,
    freeze_nn=False,
    pretrained_nn=None,
)
print("Train phase complete.")

# ---- TEST PHASE ----
# Fresh key derived from seed and fold_idx so it's reproducible but different from train
print(f"Test phase: {len(test_idx)} cells")
master_key, test_phase_key = jax.random.split(master_key)

test_vars, test_preds, test_nn = run_one_phase(
    phase_dir=test_dir,
    config=config,
    project_root=project_root,
    cell_indices=test_idx,
    full_nfkb=nfkb_signal,
    full_p53=true_p53_values,
    time_points=time_points,
    ude_model=ude_model,
    key=test_phase_key,
    freeze_nn=True,
    pretrained_nn=trained_nn,
)
print("Test phase complete.")

print(f"Worker done: fold={fold_idx}, seed={seed}")
