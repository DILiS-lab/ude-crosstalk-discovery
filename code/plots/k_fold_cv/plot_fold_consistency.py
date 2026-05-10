"""
Train and test relative MAE per fold (paired box plot) with UDE-full and ODE-full reference lines.

Produces one figure per dataset, saved under k_fold_cv/{dataset}/.
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

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
WIDTH  = 0.3
OFFSET = 0.18

plt.rcParams.update({
    "font.size": 6, "axes.labelsize": 6, "axes.titlesize": 6,
    "xtick.labelsize": 6, "ytick.labelsize": 6,
    "legend.fontsize": 6, "lines.linewidth": 0.5,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2, "ytick.major.size": 2,
    "patch.linewidth": 0.5,
})


def load_full_rel_mae_per_cell(exp_dir):
    true = np.loadtxt(TRUE_PATH, delimiter=",")
    vals = []
    for seed_dir in sorted(Path(exp_dir).glob("run_seed_*")):
        pred_path = seed_dir / "p53_generated.csv"
        if not pred_path.exists():
            continue
        pred = np.loadtxt(pred_path, delimiter=",")
        rel_mae = np.abs(pred - true).mean(axis=0) / np.abs(true).mean(axis=0)
        vals.append(rel_mae)
    return np.concatenate(vals)


def make_bp(ax, data, positions, color):
    bp = ax.boxplot(
        data,
        positions=positions,
        widths=WIDTH,
        patch_artist=True,
        medianprops=dict(color="k", linewidth=0.8),
        whiskerprops=dict(linewidth=0.5),
        capprops=dict(linewidth=0.5),
        flierprops=dict(marker="o", markersize=1, markeredgewidth=0.3,
                        markerfacecolor=color, markeredgecolor=color, alpha=0.4),
    )
    for patch in bp["boxes"]:
        patch.set_facecolor(color)
        patch.set_alpha(0.4)
    return bp


for dataset, cfg in DATASETS.items():
    out_dir = SCRIPT_DIR / dataset
    out_dir.mkdir(exist_ok=True)

    df = load_cv_results(cfg["kfold_dir"], reduce="per_cell_rel")
    folds = sorted(df["fold"].unique())

    train_data = [df.loc[(df["fold"] == f) & (df["phase"] == "train"), "rel_mae"].values for f in folds]
    test_data  = [df.loc[(df["fold"] == f) & (df["phase"] == "test"),  "rel_mae"].values for f in folds]

    train_positions = [f - OFFSET for f in folds]
    test_positions  = [f + OFFSET for f in folds]

    ude_median = np.median(load_full_rel_mae_per_cell(cfg["ude_dir"]))
    ode_median = np.median(load_full_rel_mae_per_cell(cfg["ode_dir"]))

    rng = np.random.default_rng(1)

    fig, ax = plt.subplots(figsize=(3.5, 2.0))

    make_bp(ax, train_data, train_positions, TRAIN_COLOR)
    make_bp(ax, test_data,  test_positions,  TEST_COLOR)

    for pos, vals, color in (list(zip(train_positions, train_data, [TRAIN_COLOR] * len(folds))) +
                              list(zip(test_positions,  test_data,  [TEST_COLOR]  * len(folds)))):
        jitter = rng.uniform(-0.06, 0.06, len(vals))
        ax.scatter(pos + jitter, vals, s=1.0, color=color, alpha=0.3, lw=0)

    ax.axhline(ude_median, color=UDE_FULL_COLOR, ls="--", lw=0.8, label="UDE full (median)")
    ax.axhline(ode_median, color=ODE_FULL_COLOR, ls="--", lw=0.8, label="ODE full (median)")

    ax.set_xticks(folds)
    ax.set_xticklabels([f"Fold {f}" for f in folds])
    ax.set_xlim(-0.5, max(folds) + 0.5)
    ax.set_ylabel("Relative MAE (per cell, over time)")
    ax.grid(axis="y", ls="-", alpha=0.1, lw=0.1, c="k")
    ax.set_title(cfg["label"])

    ax.legend(
        handles=[
            Patch(facecolor=TRAIN_COLOR,    alpha=0.5, label="Train (CV)"),
            Patch(facecolor=TEST_COLOR,     alpha=0.5, label="Test (CV)"),
            plt.Line2D([0], [0], color=UDE_FULL_COLOR, ls="--", lw=0.8, label="UDE full (median)"),
            plt.Line2D([0], [0], color=ODE_FULL_COLOR, ls="--", lw=0.8, label="ODE full (median)"),
        ],
        loc="upper right", frameon=False, ncol=2,
    )

    fig.tight_layout()
    out = out_dir / "rmse_per_fold_boxplot.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")
