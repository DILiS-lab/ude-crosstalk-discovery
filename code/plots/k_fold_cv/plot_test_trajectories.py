"""
Sample held-out cell trajectories: p53_true (black) vs seed-ensemble prediction band.

6 cells chosen near the median test relative MAE (one per fold where possible), 2×3 grid.
Produces one figure per dataset, saved under k_fold_cv/{dataset}/.
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils.cv_results import load_cv_results

CODE_DIR   = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent

DATASETS = {
    "konrath": {
        "kfold_dir": CODE_DIR / "experiments" / "ude_konrath2020_kfold_5fold_2026-05-09_170241",
        "label":     "Konrath et al. (2020)",
    },
    "hunziker": {
        "kfold_dir": CODE_DIR / "experiments" / "ude_hunziker2010_kfold_5fold_2026-05-10_075001",
        "label":     "Hunziker et al. (2010)",
    },
}

PRED_COLOR  = "C1"
TRUE_COLOR  = "k"
TIME_POINTS = np.arange(0.1, 20.1, 0.1)

plt.rcParams.update({
    "font.size": 6, "axes.labelsize": 6, "axes.titlesize": 6,
    "xtick.labelsize": 6, "ytick.labelsize": 6,
    "legend.fontsize": 6, "lines.linewidth": 0.5,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2, "ytick.major.size": 2,
    "patch.linewidth": 0.5,
})


def select_cells(test_pc, n=6):
    """Pick n cells near the median test rel MAE, one per fold where possible."""
    selected = []
    for fold in sorted(test_pc["fold"].unique()):
        fold_df      = test_pc[test_pc["fold"] == fold]
        fold_rel_mae = fold_df.groupby("cell_index")["rel_mae"].median()
        cell = (fold_rel_mae - fold_rel_mae.median()).abs().idxmin()
        selected.append((fold, cell))
        if len(selected) == n:
            return selected
    # pad if fewer than n folds
    all_cells = test_pc.groupby("cell_index")["rel_mae"].median().sort_values()
    already   = {c for _, c in selected}
    for cell_idx in all_cells.index:
        if cell_idx not in already:
            fold = test_pc.loc[test_pc["cell_index"] == cell_idx, "fold"].iloc[0]
            selected.append((fold, cell_idx))
            if len(selected) == n:
                break
    return selected


def get_cell_arrays(raw, n_tp, fold, cell_index):
    sub = raw[(raw["fold"] == fold) & (raw["cell_index"] == cell_index)]
    pred_rows = []
    for seed, grp in sub.groupby("seed"):
        pred_rows.append(grp.sort_values("timepoint")["p53_pred"].values[:n_tp])
    first_seed = sub["seed"].iloc[0]
    true_vals = sub[sub["seed"] == first_seed].sort_values("timepoint")["p53_true"].values[:n_tp]
    return true_vals, np.stack(pred_rows, axis=0)


def plot_band(ax, x, data, color, ls, label, alpha=0.25):
    mean = data.mean(axis=0)
    lo   = np.percentile(data, 5,  axis=0)
    hi   = np.percentile(data, 95, axis=0)
    ax.plot(x, mean, color=color, ls=ls, lw=0.8, label=label)
    ax.fill_between(x, lo, hi, color=color, alpha=alpha)


for dataset, cfg in DATASETS.items():
    out_dir = SCRIPT_DIR / dataset
    out_dir.mkdir(exist_ok=True)

    per_cell = load_cv_results(cfg["kfold_dir"], reduce="per_cell_rel")
    test_pc  = per_cell[per_cell["phase"] == "test"]
    selected = select_cells(test_pc)

    raw  = load_cv_results(cfg["kfold_dir"], reduce="none")
    raw  = raw[raw["phase"] == "test"]
    n_tp = raw["timepoint"].nunique()
    x    = TIME_POINTS[:n_tp]

    fig = plt.figure(figsize=(5.5, 2.8))
    fig.suptitle(cfg["label"], fontsize=6)
    gs  = GridSpec(2, 3, figure=fig, hspace=0.55, wspace=0.35)

    ax_ref = None
    for idx, (fold, cell) in enumerate(selected):
        row, col = divmod(idx, 3)
        ax = fig.add_subplot(gs[row, col], sharey=ax_ref)
        if ax_ref is None:
            ax_ref = ax

        true_vals, pred_mat = get_cell_arrays(raw, n_tp, fold, cell)

        ax.plot(x, true_vals, color=TRUE_COLOR, lw=0.8, label="True")
        plot_band(ax, x, pred_mat, PRED_COLOR, "-", "Predicted")

        ax.set_title(f"Cell {cell}  (fold {fold})")
        ax.set_xlim(x[0], x[-1])
        ax.grid(ls="-", alpha=0.1, lw=0.1, c="k")

        if col == 0:
            ax.set_ylabel("p53 (a.u.)")
        if row == 1:
            ax.set_xlabel("Time (h)")
        if idx == 0:
            ax.legend(loc="upper right", frameon=False, handlelength=1.0)

    out = out_dir / "test_trajectories_sample.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")
