"""
Learned crosstalk function: per-fold CV bands vs. full-UDE reference.

For each dataset (Konrath, Hunziker):
  - 5 colored bands: mean ± 90 % CI of the crosstalk learned across seeds
    within each fold (from fold_*/seed_*/train/learned_crosstalk_factor_values.csv)
  - 1 black band: mean ± 90 % CI across all seeds of the full UDE
    (from run_seed_*/learned_crosstalk_factor_values.csv)

Saved to k_fold_cv/{dataset}/crosstalk_kfold_vs_full.pdf
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

CODE_DIR   = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent

DATASETS = {
    "konrath": {
        "kfold_dir": CODE_DIR / "experiments" / "ude_konrath2020_kfold_5fold_2026-05-09_170241",
        "ude_dir":   CODE_DIR / "experiments" / "ude_konrath2020_uncertainty_2026-01-20_100950",
        "label":     "Konrath et al. (2020)",
    },
    "hunziker": {
        "kfold_dir": CODE_DIR / "experiments" / "ude_hunziker2010_kfold_5fold_2026-05-10_075001",
        "ude_dir":   CODE_DIR / "experiments" / "ude_hunziker2010_uncertainty_2026-01-20_181050",
        "label":     "Hunziker et al. (2010)",
    },
}

FOLD_COLORS  = ["C0", "C1", "C2", "C3", "C4"]
FULL_COLOR   = "black"
N_GRID       = 1000
ALPHA_BAND   = 0.20
ALPHA_FULL   = 0.15

plt.rcParams.update({
    "font.size": 6, "axes.labelsize": 6, "axes.titlesize": 6,
    "xtick.labelsize": 6, "ytick.labelsize": 6,
    "legend.fontsize": 6, "lines.linewidth": 0.5,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2, "ytick.major.size": 2,
    "patch.linewidth": 0.5,
})


def load_crosstalk_curves(csv_paths, n_grid=N_GRID):
    """
    Load multiple learned_crosstalk_factor_values.csv files, interpolate each
    onto a common grid, and return (x_grid, all_ys) where all_ys is (n, n_grid).
    """
    dfs = []
    x_min, x_max = float("inf"), float("-inf")

    for p in csv_paths:
        try:
            data = np.loadtxt(p, delimiter=",", skiprows=1)
            if data.ndim == 1:
                data = data.reshape(1, -1)
            x_min = min(x_min, data[:, 0].min())
            x_max = max(x_max, data[:, 0].max())
            dfs.append(data)
        except Exception as e:
            print(f"  Warning: could not read {p}: {e}")

    if not dfs:
        return None, None

    x_grid = np.linspace(x_min, x_max, n_grid)
    ys = np.array([np.interp(x_grid, d[:, 0], d[:, 1]) for d in dfs])
    return x_grid, ys


def plot_band(ax, x, ys, color, lw, alpha, label, zorder=2):
    mean = ys.mean(axis=0)
    lo   = np.percentile(ys,  5, axis=0)
    hi   = np.percentile(ys, 95, axis=0)
    ax.plot(x, mean, color=color, lw=lw, zorder=zorder, label=label)
    ax.fill_between(x, lo, hi, color=color, alpha=alpha, zorder=zorder - 1)


for dataset, cfg in DATASETS.items():
    out_dir = SCRIPT_DIR / dataset
    out_dir.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(3.5, 2.4))

    kfold_dir = Path(cfg["kfold_dir"])
    fold_dirs  = sorted(d for d in kfold_dir.glob("fold_*") if d.is_dir())
    n_folds    = len(fold_dirs)

    # ── per-fold bands ────────────────────────────────────────────────────────
    for fi, fold_dir in enumerate(fold_dirs):
        fold_id = int(fold_dir.name.split("_")[1])
        csvs = list(fold_dir.glob("seed_*/train/learned_crosstalk_factor_values.csv"))
        if not csvs:
            print(f"  No crosstalk CSVs found in {fold_dir}")
            continue

        x, ys = load_crosstalk_curves(csvs)
        if x is None:
            continue

        color = FOLD_COLORS[fi % len(FOLD_COLORS)]
        plot_band(ax, x, ys, color=color, lw=0.8, alpha=ALPHA_BAND,
                  label=f"Fold {fold_id + 1}", zorder=2)

    # ── full-UDE reference ────────────────────────────────────────────────────
    full_csvs = list(Path(cfg["ude_dir"]).glob("run_seed_*/learned_crosstalk_factor_values.csv"))
    if full_csvs:
        x_full, ys_full = load_crosstalk_curves(full_csvs)
        if x_full is not None:
            plot_band(ax, x_full, ys_full, color=FULL_COLOR, lw=1.2,
                      alpha=ALPHA_FULL, label="UDE (full data)", zorder=4)
    else:
        print(f"  No full-UDE crosstalk CSVs found for {dataset}")

    ax.set_xlabel(r"NF-$\kappa$B")
    ax.set_ylabel("Crosstalk factor")
    ax.set_title(cfg["label"])
    ax.grid(ls="-", alpha=0.1, lw=0.1, c="k")
    ax.legend(loc="best", frameon=False, ncol=2)

    fig.tight_layout()
    out = out_dir / "crosstalk_kfold_vs_full.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")
