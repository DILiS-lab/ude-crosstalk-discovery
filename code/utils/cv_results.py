"""
load_cv_results: tidy DataFrame loader for k-fold CV experiment folders.

Usage
-----
from utils.cv_results import load_cv_results

df      = load_cv_results("experiments/ude_konrath2020_kfold_5fold_...")
# columns: fold, seed, phase, cell_index, timepoint, p53_pred, p53_true, error

per_cell = load_cv_results(exp, reduce="per_cell")
# columns: fold, seed, phase, cell_index, rmse_over_time

per_time = load_cv_results(exp, reduce="per_time")
# columns: fold, seed, phase, timepoint, rmse_across_cells
"""

from pathlib import Path
import numpy as np
import pandas as pd


def load_cv_results(
    experiment_folder,
    reduce: str = "none",
) -> pd.DataFrame:
    """
    Walk fold_*/seed_*/{train,test}/ and assemble a tidy DataFrame.

    Parameters
    ----------
    experiment_folder : str or Path
    reduce : {"none", "per_cell", "per_time"}
        "none"     → full long-format (fold, seed, phase, cell_index, timepoint,
                      p53_pred, p53_true, error)
        "per_cell" → RMSE over time per (fold, seed, phase, cell_index)
        "per_time" → RMSE across cells per (fold, seed, phase, timepoint)
    """
    exp = Path(experiment_folder)
    chunks = []

    for fold_dir in sorted(exp.glob("fold_*")):
        try:
            fold_id = int(fold_dir.name.split("_")[1])
        except (IndexError, ValueError):
            continue

        for seed_dir in sorted(fold_dir.glob("seed_*")):
            try:
                seed = int(seed_dir.name.split("_")[1])
            except (IndexError, ValueError):
                continue

            for phase in ("train", "test"):
                phase_dir = seed_dir / phase
                if not phase_dir.exists():
                    continue

                ci_path = phase_dir / "cell_indices.csv"
                pred_path = phase_dir / "p53_predictions.csv"
                true_path = phase_dir / "p53_true.csv"
                err_path = phase_dir / "pointwise_error.csv"

                if not all(p.exists() for p in [ci_path, pred_path, true_path, err_path]):
                    continue

                cell_idx = np.loadtxt(ci_path, dtype=int, delimiter=",").ravel()
                preds = np.loadtxt(pred_path, delimiter=",")  # (T, C)
                trues = np.loadtxt(true_path, delimiter=",")  # (T, C)
                errors = np.loadtxt(err_path, delimiter=",")  # (T, C)

                if preds.ndim == 1:
                    preds = preds.reshape(-1, 1)
                    trues = trues.reshape(-1, 1)
                    errors = errors.reshape(-1, 1)

                T, C = preds.shape
                timepoints = np.arange(T)

                # Build long-format: C cells × T timepoints = C*T rows
                # cell_idx[i] repeats T times; timepoints repeat for each cell
                ci_rep = np.repeat(cell_idx, T)       # C*T
                tp_rep = np.tile(timepoints, C)        # C*T

                chunks.append(
                    pd.DataFrame(
                        {
                            "fold": fold_id,
                            "seed": seed,
                            "phase": phase,
                            "cell_index": ci_rep,
                            "timepoint": tp_rep,
                            "p53_pred": preds.T.ravel(),
                            "p53_true": trues.T.ravel(),
                            "error": errors.T.ravel(),
                        }
                    )
                )

    if not chunks:
        return pd.DataFrame(
            columns=[
                "fold", "seed", "phase", "cell_index",
                "timepoint", "p53_pred", "p53_true", "error",
            ]
        )

    df = pd.concat(chunks, ignore_index=True)

    if reduce == "none":
        return df

    elif reduce == "per_cell":
        return (
            df.groupby(["fold", "seed", "phase", "cell_index"], sort=False)
            .apply(lambda g: np.sqrt((g["error"] ** 2).mean()), include_groups=False)
            .reset_index(name="rmse_over_time")
        )

    elif reduce == "per_time":
        return (
            df.groupby(["fold", "seed", "phase", "timepoint"], sort=False)
            .apply(lambda g: np.sqrt((g["error"] ** 2).mean()), include_groups=False)
            .reset_index(name="rmse_across_cells")
        )

    else:
        raise ValueError(
            f"reduce must be 'none', 'per_cell', or 'per_time'; got {reduce!r}"
        )
