"""
Relative MAE per timepoint: train vs test vs UDE full vs ODE full, with uncertainty bands.

Produces one figure per dataset, saved under k_fold_cv/{dataset}/.
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils.cv_results import load_cv_results

CODE_DIR   = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
TRUE_PATH  = CODE_DIR / "real_data" / "p53intTNFGamma200framesDec2025.csv"

DATASETS = {
    "konrath": {
        "kfold_dir": CODE_DIR / "experiments" / "ude_konrath2020_kfold_5fold_2026-05-09_170241",
        "ude_dir":   CODE_DIR / "experiments" / "ude_konrath2020_uncertainty_2026-01-20_100950",
        "ode_dir":   CODE_DIR / "experiments" / "ode_konrath2020_uncertainty_2026-01-20_142837",
        "label":     "Konrath et al. (2020)",
    },
    "hunziker": {
        "kfold_dir": CODE_DIR / "experiments" / "ude_hunziker2010_kfold_5fold_2026-05-10_075001",
        "ude_dir":   CODE_DIR / "experiments" / "ude_hunziker2010_uncertainty_2026-01-20_181050",
        "ode_dir":   CODE_DIR / "experiments" / "ode_hunziker2010_uncertainty_2026-01-19_132644",
        "label":     "Hunziker et al. (2010)",
    },
}

TRAIN_COLOR    = "C0"
TEST_COLOR     = "C1"
UDE_FULL_COLOR = "C2"
ODE_FULL_COLOR = "C3"
TIME_POINTS    = np.arange(0.1, 20.1, 0.1)

plt.rcParams.update({
    "font.size": 6, "axes.labelsize": 6, "axes.titlesize": 6,
    "xtick.labelsize": 6, "ytick.labelsize": 6,
    "legend.fontsize": 6, "lines.linewidth": 0.5,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2, "ytick.major.size": 2,
    "patch.linewidth": 0.5,
})


def load_full_rel_mae_per_time(exp_dir):
    """Return (n_seeds, T) matrix of per-timepoint relative MAE."""
    true = np.loadtxt(TRUE_PATH, delimiter=",")  # (T, C)
    rows = []
    for seed_dir in sorted(Path(exp_dir).glob("run_seed_*")):
        pred_path = seed_dir / "p53_generated.csv"
        if not pred_path.exists():
            continue
        pred = np.loadtxt(pred_path, delimiter=",")
        rel_mae = np.abs(pred - true).mean(axis=1) / np.abs(true).mean(axis=1)
        rows.append(rel_mae)
    return np.stack(rows, axis=0)  # (n_seeds, T)


def plot_band(ax, x, data, color, ls, label, alpha=0.20):
    mean = data.mean(axis=0)
    lo   = np.percentile(data, 5,  axis=0)
    hi   = np.percentile(data, 95, axis=0)
    ax.plot(x, mean, color=color, ls=ls, lw=1, label=label)
    ax.fill_between(x, lo, hi, color=color, alpha=alpha)


def phase_matrix(df, phase):
    sub = df[df["phase"] == phase].sort_values(["fold", "seed", "timepoint"])
    rows = []
    for (fold, seed), grp in sub.groupby(["fold", "seed"]):
        rows.append(grp.sort_values("timepoint")["rel_mae"].values)
    return np.stack(rows, axis=0)


for dataset, cfg in DATASETS.items():
    out_dir = SCRIPT_DIR / dataset
    out_dir.mkdir(exist_ok=True)

    df = load_cv_results(cfg["kfold_dir"], reduce="per_time_rel")
    n_timepoints = df["timepoint"].nunique()
    x = TIME_POINTS[:n_timepoints]

    train_mat    = phase_matrix(df, "train")
    test_mat     = phase_matrix(df, "test")
    ude_full_mat = load_full_rel_mae_per_time(cfg["ude_dir"])[:, :n_timepoints]
    ode_full_mat = load_full_rel_mae_per_time(cfg["ode_dir"])[:, :n_timepoints]

    fig, ax = plt.subplots(figsize=(3.0, 2.0))

    plot_band(ax, x, train_mat,    TRAIN_COLOR,    "-", "Train (CV)")
    plot_band(ax, x, test_mat,     TEST_COLOR,     "-", "Test (CV)")
    plot_band(ax, x, ude_full_mat, UDE_FULL_COLOR, "-", "UDE (full data)")
    plot_band(ax, x, ode_full_mat, ODE_FULL_COLOR, "-", "ODE (full data)")

    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Relative MAE across cells")
    ax.grid(ls="-", alpha=0.1, lw=0.1, c="k")
    ax.legend(loc="upper right", frameon=False)
    ax.set_ylim(0, 1)
    ax.set_title(cfg["label"])

    fig.tight_layout()
    out = out_dir / "rmse_over_time_train_vs_test.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")
