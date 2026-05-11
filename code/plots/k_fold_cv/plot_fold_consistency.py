"""
Six-violin per fold comparison:
  (UDE-kfold | UDE-full | ODE-full) × (train-cells | test-cells)

UDE-kfold is retrained per fold; UDE-full and ODE-full are trained on
the complete dataset, but their per-cell errors are split by the
fold's train/test cell membership — an equivalent oracle comparison.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
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
        "label":     "Konrath",
    },
    "hunziker": {
        "kfold_dir": CODE_DIR / "experiments" / "ude_hunziker2010_kfold_5fold_2026-05-10_075001",
        "ude_dir":   CODE_DIR / "experiments" / "ude_hunziker2010_uncertainty_2026-01-20_181050",
        "ode_dir":   CODE_DIR / "experiments" / "ode_hunziker2010_uncertainty_2026-01-19_132644",
        "label":     "Hunziker",
    },
}

# Colors by model; phase distinguished by hatch
KFOLD_COLOR    = "C0"
UDE_FULL_COLOR = "C2"
ODE_FULL_COLOR = "C3"

FOLD_SPACING   = 2.0     # distance between fold centres
VIOLIN_SPACING = 0.28    # centre-to-centre within a fold (6 violins → span 1.4)
VIOLIN_WIDTH   = 0.22
HATCH_TEST     = "////"

plt.rcParams.update({
    "font.size": 6, "axes.labelsize": 6, "axes.titlesize": 6,
    "xtick.labelsize": 6, "ytick.labelsize": 6,
    "legend.fontsize": 6, "lines.linewidth": 0.1,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2, "ytick.major.size": 2,
    "patch.linewidth": 0.5,
})


def load_full_rel_mae_matrix(exp_dir, true):
    """
    Return a (n_seeds, C) array where entry [s, c] is the relative MAE
    of seed s on cell c over all timepoints.
    """
    true = np.atleast_2d(true)       # ensure (T, C)
    denom = np.abs(true).mean(axis=0)  # (C,)
    mats = []
    for seed_dir in sorted(Path(exp_dir).glob("run_seed_*")):
        pred_path = seed_dir / "p53_generated.csv"
        if not pred_path.exists():
            continue
        pred = np.loadtxt(pred_path, delimiter=",")
        rel_mae = np.abs(pred - true).mean(axis=0) / denom  # (C,)
        mats.append(rel_mae)
    return np.array(mats)   # (n_seeds, C)


def make_bp(ax, data, pos, color, hatch=None):
    bp = ax.boxplot(
        data,
        positions=[pos],
        widths=VIOLIN_WIDTH,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="k", linewidth=0.8),
        whiskerprops=dict(linewidth=0.5),
        capprops=dict(linewidth=0.5),
    )
    for patch in bp["boxes"]:
        patch.set_facecolor(color)
        patch.set_alpha(0.45)
        if hatch:
            patch.set_hatch(hatch)
    return bp


true_global = np.loadtxt(TRUE_PATH, delimiter=",")   # (T, C_total)

for dataset, cfg in DATASETS.items():
    out_dir = SCRIPT_DIR / dataset
    out_dir.mkdir(exist_ok=True)

    # ── k-fold CV results (UDE retrained per fold) ──────────────────────────
    df = load_cv_results(cfg["kfold_dir"], reduce="per_cell_rel")
    folds = sorted(df["fold"].unique())

    # ── fold → cell membership ───────────────────────────────────────────────
    splits = pd.read_csv(cfg["kfold_dir"] / "fold_splits.csv")
    # fold_splits.csv: cell_index, fold_id  (fold_id = fold this cell is TEST in)

    # ── full-model per-cell rel-MAE matrices (n_seeds, C) ───────────────────
    ude_mat = load_full_rel_mae_matrix(cfg["ude_dir"], true_global)
    ode_mat = load_full_rel_mae_matrix(cfg["ode_dir"], true_global)

    fig, ax = plt.subplots(figsize=(7.5, 2.2))

    fold_centres = [f * FOLD_SPACING for f in range(len(folds))]
    # within-fold offsets for 6 violins (train/test × 3 models)
    offsets = np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]) * VIOLIN_SPACING

    for fi, (f, cx) in enumerate(zip(folds, fold_centres)):
        test_cells  = splits.loc[splits["fold_id"] == f,  "cell_index"].values
        train_cells = splits.loc[splits["fold_id"] != f, "cell_index"].values

        # UDE-kfold (retrained) — train and test sets
        kf_train = df.loc[(df["fold"] == f) & (df["phase"] == "train"), "rel_mae"].values
        kf_test  = df.loc[(df["fold"] == f) & (df["phase"] == "test"),  "rel_mae"].values

        # UDE-full sliced to fold membership
        ude_train = ude_mat[:, train_cells].ravel()
        ude_test  = ude_mat[:, test_cells].ravel()

        # ODE-full sliced to fold membership
        ode_train = ode_mat[:, train_cells].ravel()
        ode_test  = ode_mat[:, test_cells].ravel()

        groups = [
            (kf_train,  KFOLD_COLOR,    None),
            (kf_test,   KFOLD_COLOR,    HATCH_TEST),
            (ude_train, UDE_FULL_COLOR, None),
            (ude_test,  UDE_FULL_COLOR, HATCH_TEST),
            (ode_train, ODE_FULL_COLOR, None),
            (ode_test,  ODE_FULL_COLOR, HATCH_TEST),
        ]
        for (vals, color, hatch), off in zip(groups, offsets):
            make_bp(ax, vals, cx + off, color, hatch)

    ax.set_xticks(fold_centres)
    ax.set_xticklabels([f"Fold {f + 1}" for f in folds])
    ax.set_xlim(-FOLD_SPACING * 0.6, fold_centres[-1] + FOLD_SPACING * 0.6)
    ax.set_ylim(0, 0.5)
    ax.set_yticks(np.arange(0, 0.51, 0.05))
    ax.set_ylabel("Relative MAE (per cell, over time)")
    ax.grid(axis="y", ls="-", alpha=0.1, lw=0.1, c="k")
    ax.set_title(cfg["label"])

    legend_handles = [
        Patch(facecolor=KFOLD_COLOR,    alpha=0.5, label="UDE (CV train, train)"),
        Patch(facecolor=KFOLD_COLOR,    alpha=0.5, hatch=HATCH_TEST, label="UDE (CV train, test)"),
        Patch(facecolor=UDE_FULL_COLOR, alpha=0.5, label="UDE (full, train)"),
        Patch(facecolor=UDE_FULL_COLOR, alpha=0.5, hatch=HATCH_TEST, label="UDE (full, test)"),
        Patch(facecolor=ODE_FULL_COLOR, alpha=0.5, label="ODE (full, train)"),
        Patch(facecolor=ODE_FULL_COLOR, alpha=0.5, hatch=HATCH_TEST, label="ODE (full, test)"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", ncol=3,
              title="Model (trained on, evaluated on)", title_fontsize=6)

    fig.tight_layout()
    out = out_dir / "rmse_per_fold_boxplot.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")
