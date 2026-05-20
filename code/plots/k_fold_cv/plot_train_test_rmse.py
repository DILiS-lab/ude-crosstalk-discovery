"""
Per-cell relative MAE violin plot: train vs test vs UDE full dataset vs ODE full dataset.

Primary overfitting diagnostic: train ≈ test ≈ UDE-full, all clearly better than ODE-full.
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
    true = np.loadtxt(TRUE_PATH, delimiter=",")  # (T, C)
    vals = []
    for seed_dir in sorted(Path(exp_dir).glob("run_seed_*")):
        pred_path = seed_dir / "p53_generated.csv"
        if not pred_path.exists():
            continue
        pred = np.loadtxt(pred_path, delimiter=",")
        rel_mae = np.abs(pred - true).mean(axis=0) / np.abs(true).mean(axis=0)
        vals.append(rel_mae)
    return np.concatenate(vals)


for dataset, cfg in DATASETS.items():
    out_dir = SCRIPT_DIR / dataset
    out_dir.mkdir(exist_ok=True)

    df = load_cv_results(cfg["kfold_dir"], reduce="per_cell_rel")
    train_vals    = df.loc[df["phase"] == "train", "rel_mae"].values
    test_vals     = df.loc[df["phase"] == "test",  "rel_mae"].values
    ude_full_vals = load_full_rel_mae_per_cell(cfg["ude_dir"])
    ode_full_vals = load_full_rel_mae_per_cell(cfg["ode_dir"])

    all_data  = [train_vals, test_vals, ude_full_vals, ode_full_vals]
    colors    = [TRAIN_COLOR, TEST_COLOR, UDE_FULL_COLOR, ODE_FULL_COLOR]
    positions = [0, 1, 2, 3]
    labels    = ["UDE\n(Training set)", "UDE\n(Test set)", "UDE\n(Full dataset)", "ODE\n(Full dataset)"]

    rng = np.random.default_rng(0)

    fig, ax = plt.subplots(figsize=(2.8, 2.0))

    vp = ax.violinplot(all_data, positions=positions,
                       widths=0.55, showmedians=True, showextrema=False)

    for body, color in zip(vp["bodies"], colors):
        body.set_facecolor(color)
        body.set_alpha(0.35)
        body.set_edgecolor(color)
        body.set_linewidth(0.5)

    vp["cmedians"].set_color("k")
    vp["cmedians"].set_linewidth(0.8)

    for pos, vals, color in zip(positions, all_data, colors):
        jitter = rng.uniform(-0.08, 0.08, len(vals))
        ax.scatter(pos + jitter, vals, s=0.5, color=color, alpha=0.25, lw=0)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Relative MAE (per cell, over time)")
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(0, 1)
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.grid(axis="y", ls="-", alpha=0.1, lw=0.1, c="k")
    ax.set_title(cfg["label"])

    fig.tight_layout()
    out = out_dir / "rmse_train_vs_test_violin.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")
