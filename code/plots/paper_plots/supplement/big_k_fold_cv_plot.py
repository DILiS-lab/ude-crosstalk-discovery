"""
Supplementary figure: 4×5 combined grid of k-fold CV panels.

Konrath (rows 0-1) stacked above Hunziker (rows 2-3):
  Even rows — per-fold relative-MAE boxplots
  Odd rows  — per-fold learned crosstalk vs. full-UDE reference
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

CODE_DIR   = Path(__file__).resolve().parents[4] / "code"
SCRIPT_DIR = Path(__file__).resolve().parent
TRUE_PATH  = CODE_DIR / "real_data" / "p53intTNFGamma200framesDec2025.csv"

sys.path.insert(0, str(CODE_DIR))
from utils.cv_results import load_cv_results

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

KFOLD_COLOR    = "C0"
UDE_FULL_COLOR = "C2"
ODE_FULL_COLOR = "C3"
FULL_CT_COLOR  = "black"
FOLD_COLORS    = ["C0", "C1", "C2", "C3", "C4"]
HATCH_TEST     = "////"

N_FOLDS        = 5
VIOLIN_WIDTH   = 0.28
N_GRID         = 500
ALPHA_BAND     = 0.15

# x-positions for 6 boxes in the RMSE panel: 3 model groups, 2 boxes each
BOX_OFFSETS = np.array([-0.18, 0.18])   # train / test within a model group
MODEL_CTRS  = np.array([0.0, 0.75, 1.5])  # UDE-kfold / UDE-full / ODE-full

plt.rcParams.update({
    "font.size": 6, "axes.labelsize": 6, "axes.titlesize": 6,
    "xtick.labelsize": 5, "ytick.labelsize": 5,
    "legend.fontsize": 5, "lines.linewidth": 0.5,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2,    "ytick.major.size": 2,
    "patch.linewidth": 0.5,
})


# ── helpers ──────────────────────────────────────────────────────────────────

def load_full_rel_mae_matrix(exp_dir, true):
    """Return (n_seeds, C) relative-MAE matrix over all cells."""
    true   = np.atleast_2d(true)
    denom  = np.abs(true).mean(axis=0)
    mats   = []
    for seed_dir in sorted(Path(exp_dir).glob("run_seed_*")):
        pred_path = seed_dir / "p53_generated.csv"
        if not pred_path.exists():
            continue
        pred    = np.loadtxt(pred_path, delimiter=",")
        rel_mae = np.abs(pred - true).mean(axis=0) / denom
        mats.append(rel_mae)
    return np.array(mats)  # (n_seeds, C)


def load_crosstalk_curves(csv_paths, n_grid=N_GRID):
    """Load & interpolate learned_crosstalk_factor_values.csv files onto a common grid."""
    x_min, x_max = float("inf"), float("-inf")
    dfs = []
    for p in csv_paths:
        try:
            data = np.loadtxt(p, delimiter=",", skiprows=1)
            if data.ndim == 1:
                data = data.reshape(1, -1)
            x_min = min(x_min, data[:, 0].min())
            x_max = max(x_max, data[:, 0].max())
            dfs.append(data)
        except Exception as e:
            print(f"  Warning: {p}: {e}")
    if not dfs:
        return None, None
    x_grid = np.linspace(x_min, x_max, n_grid)
    ys = np.array([np.interp(x_grid, d[:, 0], d[:, 1]) for d in dfs])
    return x_grid, ys


def make_bp(ax, data, pos, color, hatch=None):
    bp = ax.boxplot(
        data, positions=[pos], widths=VIOLIN_WIDTH,
        patch_artist=True, showfliers=False,
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


true_global = np.loadtxt(TRUE_PATH, delimiter=",")  # (T, C_total)

AVG_FOLDS_COLOR = "C9"
THIS_FOLD_COLOR = "C4"

fig = plt.figure(figsize=(7.00787, 6.2), constrained_layout=True)
subfigs = fig.subfigures(2, 1, hspace=0.06)

for (dataset, cfg), subfig in zip(DATASETS.items(), subfigs):

    # ── load data ────────────────────────────────────────────────────────────
    df_cv   = load_cv_results(cfg["kfold_dir"], reduce="per_cell_rel")
    splits  = pd.read_csv(cfg["kfold_dir"] / "fold_splits.csv")
    folds   = sorted(df_cv["fold"].unique())

    ude_mat = load_full_rel_mae_matrix(cfg["ude_dir"], true_global)  # (S, C)
    ode_mat = load_full_rel_mae_matrix(cfg["ode_dir"], true_global)  # (S, C)

    # full-UDE crosstalk reference (common to all fold panels)
    full_ct_csvs = list(Path(cfg["ude_dir"]).glob("run_seed_*/learned_crosstalk_factor_values.csv"))
    x_full, ys_full = load_crosstalk_curves(full_ct_csvs)

    # all-folds-pooled crosstalk (all fold × seed curves together)
    all_fold_ct_csvs = list(Path(cfg["kfold_dir"]).glob("fold_*/seed_*/train/learned_crosstalk_factor_values.csv"))
    x_avg_folds, ys_avg_folds = load_crosstalk_curves(all_fold_ct_csvs)

    subfig.suptitle(cfg["label"], fontsize=7, fontweight="bold")
    axes = subfig.subplots(2, N_FOLDS)

    for fi, fold_id in enumerate(folds):
        ax_box = axes[0, fi]
        ax_ct  = axes[1, fi]

        # ── Row 0: relative-MAE boxplots ─────────────────────────────────────
        test_cells  = splits.loc[splits["fold_id"] == fold_id,  "cell_index"].values
        train_cells = splits.loc[splits["fold_id"] != fold_id, "cell_index"].values

        kf_train = df_cv.loc[(df_cv["fold"] == fold_id) & (df_cv["phase"] == "train"), "rel_mae"].values
        kf_test  = df_cv.loc[(df_cv["fold"] == fold_id) & (df_cv["phase"] == "test"),  "rel_mae"].values
        ude_train = ude_mat[:, train_cells].ravel()
        ude_test  = ude_mat[:, test_cells].ravel()
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
        positions = [MODEL_CTRS[0] + BOX_OFFSETS[0],
                     MODEL_CTRS[0] + BOX_OFFSETS[1],
                     MODEL_CTRS[1] + BOX_OFFSETS[0],
                     MODEL_CTRS[1] + BOX_OFFSETS[1],
                     MODEL_CTRS[2] + BOX_OFFSETS[0],
                     MODEL_CTRS[2] + BOX_OFFSETS[1]]
        for (vals, color, hatch), pos in zip(groups, positions):
            make_bp(ax_box, vals, pos, color, hatch)

        ax_box.set_xticks(MODEL_CTRS)
        ax_box.set_xticklabels(["UDE\n(CV)", "UDE\n(full)", "ODE\n(full)"], fontsize=4.5)
        ax_box.set_xlim(MODEL_CTRS[0] - 0.4, MODEL_CTRS[-1] + 0.4)
        ax_box.set_ylim(0, 0.5)
        ax_box.set_yticks(np.arange(0, 0.51, 0.1))
        ax_box.grid(axis="y", ls="-", alpha=0.1, lw=0.1, c="k")
        ax_box.set_title(f"Fold {fold_id + 1}", fontsize=6, fontweight="bold")
        if fi == 0:
            ax_box.set_ylabel("Time-averaged rMAE", fontsize=6)
        else:
            ax_box.tick_params(labelleft=False)

        # ── Row 1: crosstalk per fold ─────────────────────────────────────────
        fold_dir = Path(cfg["kfold_dir"]) / f"fold_{fold_id}"
        fold_ct_csvs = list(fold_dir.glob("seed_*/train/learned_crosstalk_factor_values.csv"))
        x_fold, ys_fold = load_crosstalk_curves(fold_ct_csvs)

        if x_fold is not None:
            ax_ct.plot(x_fold, ys_fold.mean(axis=0), color=THIS_FOLD_COLOR, lw=0.8, zorder=4)
            ax_ct.fill_between(x_fold,
                               np.percentile(ys_fold,  5, axis=0),
                               np.percentile(ys_fold, 95, axis=0),
                               color=THIS_FOLD_COLOR, alpha=ALPHA_BAND, zorder=3)

        if x_avg_folds is not None:
            ax_ct.plot(x_avg_folds, ys_avg_folds.mean(axis=0),
                       color=AVG_FOLDS_COLOR, lw=0.8, zorder=2)
            ax_ct.fill_between(x_avg_folds,
                               np.percentile(ys_avg_folds,  5, axis=0),
                               np.percentile(ys_avg_folds, 95, axis=0),
                               color=AVG_FOLDS_COLOR, alpha=ALPHA_BAND, zorder=1)

        if x_full is not None:
            ax_ct.plot(x_full, ys_full.mean(axis=0), color=FULL_CT_COLOR, lw=0.8, zorder=6)
            ax_ct.fill_between(x_full,
                               np.percentile(ys_full,  5, axis=0),
                               np.percentile(ys_full, 95, axis=0),
                               color=FULL_CT_COLOR, alpha=ALPHA_BAND, zorder=5)

        ax_ct.set_xlabel(r"NF-$\kappa$B", fontsize=6)
        ax_ct.grid(ls="-", alpha=0.1, lw=0.1, c="k")
        if fi == 0:
            ax_ct.set_ylabel("Crosstalk factor", fontsize=6)

        from matplotlib.lines import Line2D
        ct_legend_handles = [
            Line2D([0], [0], color=THIS_FOLD_COLOR, lw=0.8, label=f"UDE-CV (fold {fold_id + 1})"),
            Line2D([0], [0], color=AVG_FOLDS_COLOR, lw=0.8, label="UDE-CV (avg. over all folds)"),
            Line2D([0], [0], color=FULL_CT_COLOR,   lw=0.8, label="UDE (full data)"),
        ]
        ax_ct.legend(
            handles=ct_legend_handles, loc="upper left",
            ncol=1, fontsize=4.5, columnspacing=0.5, handlelength=1.0,
            handletextpad=0.4, borderpad=0.4, framealpha=0.8,
        )

    # ── shared y-axis for bottom row (autoscaled per dataset) ────────────────
    all_ylims = [ax.get_ylim() for ax in axes[1, :]]
    ymin = min(y[0] for y in all_ylims)
    ymax = max(y[1] for y in all_ylims)
    for fi, ax in enumerate(axes[1, :]):
        ax.set_ylim(ymin, ymax)
        if fi > 0:
            ax.tick_params(labelleft=False)

    # ── shared legend placed inside the last panel of the RMSE row ───────────
    legend_handles = [
        Patch(facecolor=KFOLD_COLOR,    alpha=0.5, label="UDE (CV train, train)"),
        Patch(facecolor=KFOLD_COLOR,    alpha=0.5, hatch=HATCH_TEST, label="UDE (CV train, test)"),
        Patch(facecolor=UDE_FULL_COLOR, alpha=0.5, label="UDE (full, train)"),
        Patch(facecolor=UDE_FULL_COLOR, alpha=0.5, hatch=HATCH_TEST, label="UDE (full, test)"),
        Patch(facecolor=ODE_FULL_COLOR, alpha=0.5, label="ODE (full, train)"),
        Patch(facecolor=ODE_FULL_COLOR, alpha=0.5, hatch=HATCH_TEST, label="ODE (full, test)"),
    ]
    leg = axes[0, -1].legend(
        handles=legend_handles, loc="upper right",
        ncol=3, fontsize=4.5, columnspacing=0.5, handlelength=1.0,
        handletextpad=0.4, borderpad=0.4, framealpha=0.8,
        title="Model (trained on, evaluated on)", title_fontsize=4.5,
    )
    leg.set_in_layout(False)


out_path = SCRIPT_DIR / "kfold_cv_combined.pdf"
fig.savefig(out_path)
plt.close(fig)
print(f"Saved {out_path}")
