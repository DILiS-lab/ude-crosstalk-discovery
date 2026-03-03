import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths & config
# ---------------------------------------------------------------------------
project_root = Path(__file__).resolve().parents[4]
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

TRUE_P53_PATH  = code_root / "real_data" / "p53intTNFGamma200framesDec2025.csv"
TIME_POINTS    = np.arange(0.1, 20.1, 0.1)   # 200 points, 0.1 h spacing

SAMPLE_INDICES  = (24, 47, 10, 35)
DATASETS_BY_ROW = ["konrath", "hunziker"]
DATASET_LABELS  = {"hunziker": "Hunziker", "konrath": "Konrath"}

UDE_COLOR        = {"hunziker": "C1", "konrath": "C0"}
ODE_COLOR        = {"hunziker": "C1", "konrath": "C0"}
MODEL_LS         = {"UDE": "-",  "ODE": ":"}
CROSSTALK_COLOR  = {"hunziker": "C0", "konrath": "C1"}
SR_FUNCS = {
    "hunziker": (r"$0.274\,[NF\kappa B] + 0.717$",
                 lambda x: 0.27438045 * x + 0.7170913),
    "konrath":  (r"$[NF\kappa B]^2 + 0.076\,[NF\kappa B] + 1.645$",
                 lambda x: (x + 0.076078266) * x + 1.6450808),
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _agg(paths, loader):
    arrays = [loader(p) for p in paths if p.exists()]
    arrays = [a for a in arrays if a is not None]
    if not arrays:
        return None
    min_len = min(a.shape[0] for a in arrays)
    return np.stack([a[:min_len] for a in arrays], axis=0)


def load_p53(exp_dir: Path):
    paths = sorted(exp_dir.glob("run_seed_*/p53_generated.csv"))
    return _agg(paths, lambda p: np.loadtxt(p, delimiter=","))


def load_rel_mae_per_timepoint(exp_dir: Path, true_p53: np.ndarray):
    """Compute relative MAE per timepoint across all seeds.

    rel_mae[t] = mean_cells(|pred[t] - true[t]|) / mean_cells(|true[t]|)

    Returns array of shape (n_seeds, n_timepoints).
    """
    paths = sorted(exp_dir.glob("run_seed_*/p53_generated.csv"))
    result = []
    for p in paths:
        if not p.exists():
            continue
        try:
            pred = np.loadtxt(p, delimiter=",")            # (T, C)
            n_tp = min(pred.shape[0], true_p53.shape[0])
            pred = pred[:n_tp, :]
            true = true_p53[:n_tp, :]
            # per-timepoint mean absolute error across cells
            mae_t = np.mean(np.abs(pred - true), axis=1)  # (T,)
            # normalise by mean absolute true value per timepoint
            denom = np.mean(np.abs(true), axis=1)          # (T,)
            result.append(mae_t / denom)
        except Exception:
            continue
    if not result:
        return None
    return np.stack(result, axis=0)   # (n_seeds, T)


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
    x_min   = min(d["nfkb_flat"].min() for d in seed_dfs)
    x_max   = max(d["nfkb_flat"].max() for d in seed_dfs)
    x_grid  = np.linspace(x_min, x_max, n_grid)
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
# Figure
# ---------------------------------------------------------------------------
plt.rcParams.update({"font.size": 6, "axes.labelsize": 6, "axes.titlesize": 6,
                     "xtick.labelsize": 5, "ytick.labelsize": 5,
                     "legend.fontsize": 5, "lines.linewidth": 0.1})

plt.rcParams.update({"font.size": 6, "axes.labelsize": 6, "axes.titlesize": 6,
                     "xtick.labelsize": 5, "ytick.labelsize": 5,
                     "legend.fontsize": 5, "lines.linewidth": 0.1,
                     "axes.linewidth": 0.5,
                     "xtick.major.width": 0.5, "ytick.major.width": 0.5,
                     "xtick.major.size": 2,   "ytick.major.size": 2})

grid_keywords = {"ls": "-", "alpha": 0.1, "lw": 0.1, 'c': 'k'}

true_p53 = np.loadtxt(TRUE_P53_PATH, delimiter=",")   # (200, 106)

fig = plt.figure(figsize=(7.00787, 2.5))
gs  = GridSpec(2, 5, figure=fig)

ax_rmae_konrath  = fig.add_subplot(gs[0, 0])
ax_rmae_hunziker = fig.add_subplot(gs[1, 0])
ax_samples       = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(1, 3)]
ax_crosstalk     = fig.add_subplot(gs[:, -2:])

# ---------------------------------------------------------------------------
# Relative MAE panels
# ---------------------------------------------------------------------------
for dataset, ax in zip(DATASETS_BY_ROW, [ax_rmae_konrath, ax_rmae_hunziker]):
    for model, exp_dir in RUNS[dataset].items():
        data = load_rel_mae_per_timepoint(exp_dir, true_p53)
        if data is None:
            continue
        n_tp  = data.shape[1]
        color = UDE_COLOR[dataset] if model == "UDE" else ODE_COLOR[dataset]
        plot_band(ax, TIME_POINTS[:n_tp], data,
                  color=color, ls=MODEL_LS[model], label=model)

    ax.set_xlim(0, 20)
    ax.set_ylim(bottom=0, top=1)
    ax.set_ylabel("Relative MAE")
    ax.grid(True, **grid_keywords)
    ax.legend(loc="upper right", title=DATASET_LABELS[dataset], title_fontsize=5)
    if ax is ax_rmae_hunziker:
        ax.set_xlabel("Time [h]")
    else:
        ax.tick_params(labelbottom=False)

# ---------------------------------------------------------------------------
# Sample trajectory panels  (unchanged)
# ---------------------------------------------------------------------------
ude_p53 = {dataset: load_p53(RUNS[dataset]["UDE"]) for dataset in ["konrath", "hunziker"]}

for panel_idx, sample_idx in enumerate(SAMPLE_INDICES):
    ax = ax_samples[panel_idx]

    ax.plot(TIME_POINTS, true_p53[:, sample_idx],
            color="k", lw=0.8, alpha=0.7, label="Data")

    for dataset in ["konrath", "hunziker"]:
        data = ude_p53[dataset]
        if data is None:
            continue
        cell = data[:, :, sample_idx]
        n_tp = cell.shape[1]
        plot_band(ax, TIME_POINTS[:n_tp], cell,
                  color=UDE_COLOR[dataset], ls="-", label=DATASET_LABELS[dataset])

    ax.set_xlim(0, 20)
    ax.set_ylim(bottom=0)
    ax.grid(True, **grid_keywords)

    if panel_idx >= 2:
        ax.set_xlabel("Time [h]")
    else:
        ax.tick_params(labelbottom=False)

    ax.set_ylabel("p53")

ax_samples[0].legend(loc="upper left")

# ---------------------------------------------------------------------------
# Crosstalk panel  (unchanged)
# ---------------------------------------------------------------------------
for dataset in ["hunziker", "konrath"]:
    exp_dir = RUNS[dataset]["UDE"]
    color   = UDE_COLOR[dataset]
    label   = DATASET_LABELS[dataset]

    x_grid, ys = load_crosstalk(exp_dir)
    if x_grid is None:
        continue

    plot_band(ax_crosstalk, x_grid, ys,
              color=color, ls="-", label=f"{label} (NN)")

    sr_label, sr_fn = SR_FUNCS[dataset]
    ax_crosstalk.plot(x_grid, sr_fn(x_grid),
                      color=color, ls="--", lw=1, label=sr_label)

ax_crosstalk.set_xlabel(r"NF-$\kappa$B")
ax_crosstalk.set_ylabel("Crosstalk factor")
ax_crosstalk.legend(loc="upper left", ncol=1)
ax_crosstalk.grid(True, **grid_keywords)
ax_crosstalk.set_xlim(left=0)

# ---------------------------------------------------------------------------
# Panel labels
# ---------------------------------------------------------------------------
_panel_axes = [
    ax_rmae_konrath,   # a
    ax_rmae_hunziker,  # b
    ax_samples[0],     # c
    ax_samples[2],     # d
    ax_samples[1],     # e
    ax_samples[3],     # f
    ax_crosstalk,      # g
]
for ax, label in zip(_panel_axes, "abcdefg"):
    ax.set_title(label, loc="left", fontsize=6, fontweight="bold", pad=2)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
fig.tight_layout()
output_dir  = code_root / "plots" / "paper_plots" / "supplement"
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "real_data_big_figure_rmae.pdf"
fig.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"Saved to {output_path}")
