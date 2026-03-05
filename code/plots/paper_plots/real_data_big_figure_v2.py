from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

SCRIPT_DIR   = Path(__file__).resolve().parent
project_root = Path(__file__).resolve().parents[3]
code_root    = project_root / "code"
exp_root     = code_root / "experiments"

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

TRUE_P53_PATH   = code_root / "real_data" / "p53intTNFGamma200framesDec2025.csv"
TIME_POINTS     = np.arange(0.1, 20.1, 0.1)
DATASETS_BY_ROW = ["konrath", "hunziker"]
DATASET_LABELS  = {"hunziker": "Hunziker", "konrath": "Konrath"}
UDE_COLOR       = {"hunziker": "C1", "konrath": "C0"}
ODE_COLOR       = {"hunziker": "C1", "konrath": "C0"}
MODEL_LS        = {"UDE": "-", "ODE": ":"}
grid_keywords   = {"ls": "-", "alpha": 0.1, "lw": 0.1, "c": "k"}
SAMPLE_INDICES  = (24, 47, 10, 35, 5, 60, 80, 15, 90)  # 9 cells for the 3×3 grid

CROSSTALK_COLOR = {"hunziker": "C1", "konrath": "C0"}
SR_FUNCS = {
    "hunziker": (r"$0.274\,[NF\kappa B] + 0.717$",
                 lambda x: 0.2743812 * x + 0.7170906),
    "konrath":  (r"$[NF\kappa B]^2 + 1.698$",
                 lambda x: x**2 + 1.6979214),
}


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


def plot_band(ax, x, data, color, ls, label, alpha=0.20):
    mean = data.mean(axis=0)
    lo   = np.percentile(data, 5,  axis=0)
    hi   = np.percentile(data, 95, axis=0)
    ax.plot(x, mean, color=color, ls=ls, lw=1, label=label)
    ax.fill_between(x, lo, hi, color=color, alpha=alpha)

# ---------------------------------------------------------------------------
# Layout parameters
# ---------------------------------------------------------------------------
# Relative widths of the three column sub-figures
COL_WIDTHS = [0.4, 1.2, 0.8]

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
fig = plt.figure(figsize=(7.00787, 2.5))

gs = GridSpec(1, 3, figure=fig, width_ratios=COL_WIDTHS)

# Col 0: 2 rows
gs_col0 = gs[0, 0].subgridspec(2, 1)
ax_col0_r0 = fig.add_subplot(gs_col0[0, 0])
ax_col0_r1 = fig.add_subplot(gs_col0[1, 0])

# Col 1: 2 rows x 3 cols
gs_col1 = gs[0, 1].subgridspec(3, 3)
ax_col1 = []
ax_col1_ref = None
for r in range(3):
    row = []
    for c in range(3):
        ax = fig.add_subplot(gs_col1[r, c], sharex=ax_col1_ref)
        if ax_col1_ref is None:
            ax_col1_ref = ax
        row.append(ax)
    ax_col1.append(row)

for ax in ax_col1[0] + ax_col1[1]:
    plt.setp(ax.get_xticklabels(), visible=False)


plt.setp(ax_col0_r0.get_xticklabels(), visible=False)

# ---------------------------------------------------------------------------
# Col 0: RMAE panels
# ---------------------------------------------------------------------------
for dataset, ax in zip(DATASETS_BY_ROW, [ax_col0_r0, ax_col0_r1]):
    for model, exp_dir in RUNS[dataset].items():
        data = load_rmae(exp_dir)
        if data is None:
            continue
        n_tp  = data.shape[1]
        color = UDE_COLOR[dataset] if model == "UDE" else ODE_COLOR[dataset]
        plot_band(ax, TIME_POINTS[:n_tp], data,
                  color=color, ls=MODEL_LS[model], label=model)
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 1)
    ax.grid(True, **grid_keywords)
    ax.legend(loc="upper right", title=DATASET_LABELS[dataset], title_fontsize=6)

# ---------------------------------------------------------------------------
# Col 1: sample trajectory panels (3×3)
# ---------------------------------------------------------------------------
true_p53 = np.loadtxt(TRUE_P53_PATH, delimiter=",")
ude_p53  = {ds: load_p53(RUNS[ds]["UDE"]) for ds in ["konrath", "hunziker"]}

for panel_idx, sample_idx in enumerate(SAMPLE_INDICES):
    r, c = divmod(panel_idx, 3)
    if (r, c) == (0, 2):
        continue
    ax   = ax_col1[r][c]

    ax.plot(TIME_POINTS, true_p53[:, sample_idx] / 1000,
            color="k", lw=0.8, alpha=0.7, label="Data")

    for dataset in ["hunziker", "konrath"]:
        data = ude_p53[dataset]
        if data is None:
            continue
        cell = data[:, :, sample_idx] / 1000
        n_tp = cell.shape[1]
        plot_band(ax, TIME_POINTS[:n_tp], cell,
                  color=UDE_COLOR[dataset], ls="-", label=DATASET_LABELS[dataset])

    ax.set_xlim(0, 20)
    ax.set_ylim(bottom=0)
    ax.grid(True, **grid_keywords)

ax_col1[0][2].axis("off")
# Place legend in the hidden top-right axes area
handles, labels = ax_col1[0][0].get_legend_handles_labels()
ax_col1[0][2].legend(handles, labels, loc="upper left", ncol=1, fontsize=6)

# ---------------------------------------------------------------------------
# Axis labels
# ---------------------------------------------------------------------------
# Col 0
ax_col0_r0.set_ylabel("Relative MAE")
ax_col0_r1.set_ylabel("Relative MAE")
ax_col0_r1.set_xlabel("Time [h]")

# Col 1
for r in range(3):
    ax_col1[r][0].set_ylabel(r"p53 ($\times 10^3$)")
for ax in ax_col1[2]:
    ax.set_xlabel("Time [h]")

# Col 2: single axes
ax_col2 = fig.add_subplot(gs[0, 2])

for dataset in ["hunziker", "konrath"]:
    x_grid, ys = load_crosstalk(RUNS[dataset]["UDE"])
    if x_grid is None:
        continue
    color = CROSSTALK_COLOR[dataset]
    label = DATASET_LABELS[dataset]
    plot_band(ax_col2, x_grid, ys, color=color, ls="-", label=f"{label} (NN)")
    sr_label, sr_fn = SR_FUNCS[dataset]
    ax_col2.plot(x_grid, sr_fn(x_grid), color=color, ls="--", lw=1, label=sr_label)

ax_col2.autoscale(axis="x", tight=True)
ax_col2.grid(True, **grid_keywords)
ax_col2.legend(loc="upper left", ncol=1, fontsize=6)
ax_col2.set_xlabel(r"NF-$\kappa$B")
ax_col2.set_ylabel("Crosstalk factor")

# ---------------------------------------------------------------------------
# Panel labels
# ---------------------------------------------------------------------------
for ax, label in zip([ax_col0_r0, ax_col0_r1, ax_col1[0][0], ax_col2], "abcd"):
    ax.set_title(label, loc="left", fontsize=6, fontweight="bold", pad=2)

plt.tight_layout()
plt.savefig(SCRIPT_DIR / "real_data_big_figure_v2.pdf", dpi=300, bbox_inches="tight")
# plt.show()
