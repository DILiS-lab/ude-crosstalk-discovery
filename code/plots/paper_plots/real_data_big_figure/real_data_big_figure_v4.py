import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

SCRIPT_DIR   = Path(__file__).resolve().parent
code_root    = Path(__file__).resolve().parents[3]   # .../code/
exp_root     = code_root / "experiments"

sys.path.insert(0, str(code_root))
from utils.cv_results import load_cv_results

RUNS = {
    "hunziker": {
        "ODE": exp_root / "ode_hunziker2010_uncertainty_2026-01-19_132644",
        "UDE": exp_root / "ude_hunziker2010_uncertainty_2026-01-20_181050",
    },
    "konrath": {
        "ODE": exp_root / "ode_konrath2020_uncertainty_2026-01-20_142837",
        "UDE": exp_root / "ude_konrath2020_uncertainty_2026-01-20_100950",
    },
}

KFOLD_RUNS = {
    "konrath": {
        "kfold_dir": exp_root / "ude_konrath2020_kfold_5fold_2026-05-09_170241",
        "ude_dir":   exp_root / "ude_konrath2020_uncertainty_2026-01-20_100950",
        "ode_dir":   exp_root / "ode_konrath2020_uncertainty_2026-01-20_142837",
    },
    "hunziker": {
        "kfold_dir": exp_root / "ude_hunziker2010_kfold_5fold_2026-05-10_075001",
        "ude_dir":   exp_root / "ude_hunziker2010_uncertainty_2026-01-20_181050",
        "ode_dir":   exp_root / "ode_hunziker2010_uncertainty_2026-01-19_132644",
    },
}

TRUE_P53_PATH   = code_root / "real_data" / "p53intTNFGamma200framesDec2025.csv"
TIME_POINTS     = np.arange(0.1, 20.1, 0.1)
DATASETS_BY_ROW = ["konrath", "hunziker"]
DATASET_LABELS  = {"hunziker": "Hunziker", "konrath": "Konrath"}
UDE_COLOR       = {"hunziker": "C1", "konrath": "C0"}
ODE_COLOR       = {"hunziker": "C1", "konrath": "C0"}
MODEL_LS        = {"UDE": "-", "ODE": ":"}
grid_keywords   = {"ls": "-", "alpha": 0.1, "lw": 0.1, "c": "k"}
SAMPLE_INDICES  = (47, 24, 35, 5, 80, 15)  # 6 cells for the 3×2 grid

CROSSTALK_COLOR = {"hunziker": "C1", "konrath": "C0"}
SR_FUNCS = {
    "hunziker": (r"$0.274\,[NF\kappa B] + 0.717$",
                 lambda x: 0.2743812 * x + 0.7170906),
    "konrath":  (r"$[NF\kappa B]^2 + 1.698$",
                 lambda x: x**2 + 1.6979214),
}

CV_YLIM        = (0, 0.5)
TRAIN_COLOR    = "C2"
TEST_COLOR     = "C3"
UDE_FULL_COLOR = "C4"
ODE_FULL_COLOR = "C5"


def _agg(paths, loader):
    arrays = [loader(p) for p in paths if p.exists()]
    arrays = [a for a in arrays if a is not None]
    if not arrays:
        return None
    min_len = min(a.shape[0] for a in arrays)
    return np.stack([a[:min_len] for a in arrays], axis=0)


def load_rmae(exp_dir: Path):
    true_data = np.loadtxt(TRUE_P53_PATH, delimiter=",")
    denom     = np.mean(np.abs(true_data), axis=1)
    def compute_rmae(p):
        pred = np.loadtxt(p, delimiter=",")
        mae  = np.mean(np.abs(true_data - pred), axis=1)
        return mae / denom
    paths = sorted(exp_dir.glob("run_seed_*/p53_generated.csv"))
    return _agg(paths, compute_rmae)


def load_p53(exp_dir: Path):
    paths = sorted(exp_dir.glob("run_seed_*/p53_generated.csv"))
    return _agg(paths, lambda p: np.loadtxt(p, delimiter=","))


def load_crosstalk(exp_dir: Path, n_grid: int = 500):
    import pandas as pd
    seed_dfs = []
    for csv in sorted(exp_dir.glob("run_seed_*/learned_crosstalk_factor_values.csv")):
        try:
            df = pd.read_csv(csv).sort_values("nfkb_flat")
            seed_dfs.append(df)
        except Exception:
            continue
    if not seed_dfs:
        return None, None
    x_min = min(d["nfkb_flat"].min() for d in seed_dfs)
    x_max = max(d["nfkb_flat"].max() for d in seed_dfs)
    x_grid = np.linspace(x_min, x_max, n_grid)
    ys = np.stack([
        np.interp(x_grid, d["nfkb_flat"].values, d["pred_synth_factor"].values)
        for d in seed_dfs
    ])
    return x_grid, ys


def load_full_rel_mae_per_cell(exp_dir):
    true = np.loadtxt(TRUE_P53_PATH, delimiter=",")
    vals = []
    for seed_dir in sorted(Path(exp_dir).glob("run_seed_*")):
        pred_path = seed_dir / "p53_generated.csv"
        if not pred_path.exists():
            continue
        pred = np.loadtxt(pred_path, delimiter=",")
        rel_mae = np.abs(pred - true).mean(axis=0) / np.abs(true).mean(axis=0)
        vals.append(rel_mae)
    return np.concatenate(vals)


def plot_band(ax, x, data, color, ls, label, alpha=0.20, zorder=None):
    mean = data.mean(axis=0)
    lo   = np.percentile(data, 5,  axis=0)
    hi   = np.percentile(data, 95, axis=0)
    kw = {} if zorder is None else {"zorder": zorder}
    ax.plot(x, mean, color=color, ls=ls, lw=1, label=label, **kw)
    ax.fill_between(x, lo, hi, color=color, alpha=alpha, **kw)


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.size": 6, "axes.labelsize": 6, "axes.titlesize": 6,
    "xtick.labelsize": 6, "ytick.labelsize": 6,
    "legend.fontsize": 6, "lines.linewidth": 0.1,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2,    "ytick.major.size": 2,
    "patch.linewidth": 0.5,
})

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
COL_WIDTHS = [0.4, 0.4, 0.85, 0.72]

fig = plt.figure(figsize=(7.5, 2.5))
gs = GridSpec(1, 4, figure=fig, width_ratios=COL_WIDTHS)

# Col 0: CV violin panels (2 rows)
gs_col0 = gs[0, 0].subgridspec(2, 1)
ax_cv = [fig.add_subplot(gs_col0[i, 0]) for i in range(2)]
plt.setp(ax_cv[0].get_xticklabels(), visible=False)

# Col 1: RMAE panels (2 rows)
gs_col1 = gs[0, 1].subgridspec(2, 1)
ax_col1_r0 = fig.add_subplot(gs_col1[0, 0])
ax_col1_r1 = fig.add_subplot(gs_col1[1, 0])
plt.setp(ax_col1_r0.get_xticklabels(), visible=False)

# Col 2: trajectory panels (3×2)
gs_col2 = gs[0, 2].subgridspec(3, 2)
ax_col2 = []
ax_col2_ref = None
for r in range(3):
    row = []
    for c in range(2):
        ax = fig.add_subplot(gs_col2[r, c], sharex=ax_col2_ref)
        if ax_col2_ref is None:
            ax_col2_ref = ax
        row.append(ax)
    ax_col2.append(row)

for ax in ax_col2[0] + ax_col2[1]:
    plt.setp(ax.get_xticklabels(), visible=False)

# ---------------------------------------------------------------------------
# Col 0: CV violin panels
# ---------------------------------------------------------------------------
rng = np.random.default_rng(0)
cv_xtick_labels = ["Train", "Test", "Full", "Full"]
CV_ALPHAS = [0.45, 0.45, 0.45, 0.45]

for dataset, ax in zip(DATASETS_BY_ROW, ax_cv):
    cfg = KFOLD_RUNS[dataset]
    df = load_cv_results(cfg["kfold_dir"], reduce="per_cell_rel")
    train_vals    = df.loc[df["phase"] == "train", "rel_mae"].values
    test_vals     = df.loc[df["phase"] == "test",  "rel_mae"].values
    ude_full_vals = load_full_rel_mae_per_cell(cfg["ude_dir"])
    ode_full_vals = load_full_rel_mae_per_cell(cfg["ode_dir"])

    all_data = [train_vals, test_vals, ude_full_vals, ode_full_vals]
    clipped  = [np.clip(v, *CV_YLIM) for v in all_data]

    ds_color = UDE_COLOR[dataset]
    cv_colors = [ds_color, ds_color, ds_color, "gray"]

    vp = ax.violinplot(clipped, positions=[0, 1, 2, 3],
                       widths=0.55, showmedians=True, showextrema=False)

    for body, color, alpha in zip(vp["bodies"], cv_colors, CV_ALPHAS):
        body.set_facecolor(color)
        body.set_alpha(alpha)
        body.set_edgecolor(color)
        body.set_linewidth(0.5)

    vp["cmedians"].set_color("k")
    vp["cmedians"].set_linewidth(0.8)

    for pos, vals, color in zip([0, 1, 2, 3], clipped, cv_colors):
        jitter = rng.uniform(-0.08, 0.08, len(vals))
        ax.scatter(pos + jitter, vals, s=0.5, color=color, alpha=0.25, lw=0)

    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(cv_xtick_labels)
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(*CV_YLIM)
    ax.set_yticks(np.arange(CV_YLIM[0], CV_YLIM[1] + 0.1, 0.1))
    ax.grid(axis="y", **grid_keywords)
    ax.set_ylabel("Relative MAE")
    legend_handles = [
        mpatches.Patch(facecolor=ds_color, alpha=0.65, label="UDE"),
        mpatches.Patch(facecolor="gray",   alpha=0.35, label="ODE"),
    ]
    ax.legend(handles=legend_handles, title=DATASET_LABELS[dataset], title_fontsize=6,
              loc="upper left", fontsize=6)

# ---------------------------------------------------------------------------
# Col 1: RMAE panels
# ---------------------------------------------------------------------------
for dataset, ax in zip(DATASETS_BY_ROW, [ax_col1_r0, ax_col1_r1]):
    for model, exp_dir in RUNS[dataset].items():
        data = load_rmae(exp_dir)
        if data is None:
            continue
        n_tp  = data.shape[1]
        color = UDE_COLOR[dataset] if model == "UDE" else "gray"
        plot_band(ax, TIME_POINTS[:n_tp], data,
                  color=color, ls=MODEL_LS[model], label=model)
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 1)
    ax.grid(True, **grid_keywords)
    ax.legend(loc="upper right", title=DATASET_LABELS[dataset], title_fontsize=6)

# ---------------------------------------------------------------------------
# Col 2: sample trajectory panels (3×2)
# ---------------------------------------------------------------------------
true_p53 = np.loadtxt(TRUE_P53_PATH, delimiter=",")
ude_p53  = {ds: load_p53(RUNS[ds]["UDE"]) for ds in ["konrath", "hunziker"]}


def _rel_mae(dataset, sample_idx):
    data = ude_p53[dataset]
    if data is None:
        return None, None
    n_tp = data.shape[1]
    true_cell = true_p53[:n_tp, sample_idx]
    denom = np.mean(np.abs(true_cell))
    if denom == 0:
        return None, None
    mae_seeds = np.mean(np.abs(true_cell[np.newaxis, :] - data[:, :, sample_idx]), axis=1) / denom
    return float(np.mean(mae_seeds)), float(np.std(mae_seeds))


for panel_idx, sample_idx in enumerate(SAMPLE_INDICES):
    r, c = divmod(panel_idx, 2)
    ax   = ax_col2[r][c]

    ax.plot(TIME_POINTS, true_p53[:, sample_idx] / 1000,
            color="k", lw=0.8, alpha=0.7, label="Data")

    for dataset in ["konrath", "hunziker"]:
        data = ude_p53[dataset]
        if data is None:
            continue
        cell = data[:, :, sample_idx] / 1000
        n_tp = cell.shape[1]
        zorder = 3 if dataset == "konrath" else 2
        plot_band(ax, TIME_POINTS[:n_tp], cell,
                  color=UDE_COLOR[dataset], ls="-", label=DATASET_LABELS[dataset],
                  zorder=zorder)

    ax.set_xlim(0, 20)
    ax.set_ylim(bottom=0)
    ax.grid(True, **grid_keywords)

    for i, ds in enumerate(["konrath", "hunziker"]):
        mean, std = _rel_mae(ds, sample_idx)
        if mean is not None:
            prefix = "rMAE=" if (r, c) == (0, 0) else ""
            ax.text(0.95, 0.20 - i * 0.13, f"{prefix}{mean:.3f}±{std:.3f}",
                    transform=ax.transAxes, fontsize=4,
                    color=UDE_COLOR[ds], va="bottom", ha="right")

handles, labels = ax_col2[0][0].get_legend_handles_labels()
ax_col2[0][1].legend(handles, labels, loc="upper left", ncol=1, fontsize=6)

# ---------------------------------------------------------------------------
# Axis labels
# ---------------------------------------------------------------------------
# Col 0 (CV)
ax_cv[1].set_xlabel("Dataset")

# Col 1 (RMAE)
ax_col1_r0.set_ylabel("Relative MAE")
ax_col1_r1.set_ylabel("Relative MAE")
ax_col1_r1.set_xlabel("Time [h]")

# Col 2 (trajectories)
for r in range(3):
    ax_col2[r][0].set_ylabel(r"p53 ($\times 10^3$)")
for c in range(2):
    ax_col2[2][c].set_xlabel("Time [h]")

# Col 3: crosstalk
ax_col3 = fig.add_subplot(gs[0, 3])

for dataset in ["konrath", "hunziker"]:
    x_grid, ys = load_crosstalk(RUNS[dataset]["UDE"])
    if x_grid is None:
        continue
    color = CROSSTALK_COLOR[dataset]
    label = DATASET_LABELS[dataset]
    plot_band(ax_col3, x_grid, ys, color=color, ls="-", label=f"{label} (NN)")
    sr_label, sr_fn = SR_FUNCS[dataset]
    ax_col3.plot(x_grid, sr_fn(x_grid), color=color, ls="--", lw=1, label=sr_label)

ax_col3.autoscale(axis="x", tight=True)
ax_col3.grid(True, **grid_keywords)
ax_col3.legend(loc="upper left", ncol=1, fontsize=6)
ax_col3.set_xlabel(r"NF-$\kappa$B")
ax_col3.set_ylabel("Crosstalk factor")

# ---------------------------------------------------------------------------
# Panel labels
# ---------------------------------------------------------------------------
for ax, label in zip([ax_cv[0], ax_col1_r0, ax_col2[0][0], ax_col3], "abcd"):
    ax.set_title(label, loc="left", fontsize=6, fontweight="bold", pad=2)

plt.tight_layout()
plt.savefig(SCRIPT_DIR / "real_data_big_figure_v4.pdf", dpi=300, bbox_inches="tight")
# plt.show()
