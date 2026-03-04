import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths & config
# ---------------------------------------------------------------------------
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

TRUE_P53_PATH  = code_root / "real_data" / "p53intTNFGamma200framesDec2025.csv"
TIME_POINTS    = np.arange(0.1, 20.1, 0.1)   # 200 points, 0.1 h spacing

SAMPLE_INDICES  = (24, 47, 10, 35)            # four single cells shown across panels
DATASETS_BY_ROW = ["konrath", "hunziker"]     # row 0 = konrath, row 1 = hunziker
DATASET_LABELS  = {"hunziker": "Hunziker", "konrath": "Konrath"}

UDE_COLOR   = {"hunziker": "C1", "konrath": "C0"}
ODE_COLOR   = {"hunziker": "C1", "konrath": "C0"}#"k"
MODEL_LS    = {"UDE": "-",  "ODE": ":"}

CROSSTALK_COLOR = {"hunziker": "C0", "konrath": "C1"}
SR_FUNCS = {
    "hunziker": (r"$0.274\,[NF\kappa B] + 0.717$",
                 lambda x: 0.27438045 * x + 0.7170913),
    "konrath":  (r"$[NF\kappa B]^2 + 0.076\,[NF\kappa B] + 1.645$",
                 lambda x: (x + 0.076078266) * x + 1.6450808),
}

# Hill-form symbolic regression results
# Konrath: f(x) = x  →  offset + scale * x^n / (Kd^n + x^n)
# Hunziker: f(x) = x² →  offset + scale * (x²)^n / (Kd^n + (x²)^n)
SR_HILL_FUNCS = {
    "konrath": (
        r"$1.68 + 2.04\,\frac{x^{1.77}}{0.99^{1.77}+x^{1.77}}$",
        lambda x: 1.676529 + 2.035762 * x**1.773329 / (0.996679**1.773329 + x**1.773329),
    ),
    "hunziker": (
        r"$0.76 + 0.29\,\frac{(x^2)^{1.52}}{0.44^{1.52}+(x^2)^{1.52}}$",
        lambda x: 0.764203 + 0.289357 * (x**2)**1.515626 / (0.440040**1.515626 + (x**2)**1.515626),
    ),
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


def load_rmae(exp_dir: Path):
    true_data = np.loadtxt(TRUE_P53_PATH, delimiter=",")   # (T, N_cells)
    denom     = np.mean(np.abs(true_data), axis=1)         # (T,)

    def compute_rmae(p):
        pred = np.loadtxt(p, delimiter=",")                # (T, N_cells)
        mae  = np.mean(np.abs(true_data - pred), axis=1)  # (T,)
        return mae / denom                                 # (T,)

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

fig = plt.figure(figsize=(7.00787, 2.5))
gs  = GridSpec(2, 5, figure=fig,)# wspace=0.45, hspace=0.45)

ax_rmse_konrath  = fig.add_subplot(gs[0, 0])
ax_rmse_hunziker = fig.add_subplot(gs[1, 0])
ax_samples       = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(1, 3)]
ax_crosstalk     = fig.add_subplot(gs[:, -2:])

# ---------------------------------------------------------------------------
# RMSE panels
# ---------------------------------------------------------------------------
for dataset, ax in zip(DATASETS_BY_ROW, [ax_rmse_konrath, ax_rmse_hunziker]):
    for model, exp_dir in RUNS[dataset].items():
        data = load_rmae(exp_dir)
        if data is None:
            continue
        n_tp = data.shape[1]
        color = UDE_COLOR[dataset] if model == "UDE" else ODE_COLOR[dataset]
        plot_band(ax, TIME_POINTS[:n_tp], data,
                  color=color, ls=MODEL_LS[model], label=model)

    ax.set_xlim(0, 20)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Relative MAE")
    #ax.set_ylim(bottom=0, top=1)
    ax.grid(True, **grid_keywords)
    ax.legend(loc="upper right", title=DATASET_LABELS[dataset], title_fontsize=5)
    if ax is ax_rmse_hunziker:
        ax.set_xlabel("Time [h]")
    else:
        ax.tick_params(labelbottom=False)

# ---------------------------------------------------------------------------
# Sample trajectory panels
# ---------------------------------------------------------------------------
true_p53 = np.loadtxt(TRUE_P53_PATH, delimiter=",")   # (200, 106)

# Pre-load UDE predictions for both datasets
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
# Crosstalk panel
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

    sr_hill_label, sr_hill_fn = SR_HILL_FUNCS[dataset]
    ax_crosstalk.plot(x_grid, sr_hill_fn(x_grid),
                      color=color, ls=":", lw=1, label=sr_hill_label)

ax_crosstalk.set_xlabel(r"NF-$\kappa$B")
ax_crosstalk.set_ylabel("Crosstalk factor")
ax_crosstalk.legend(loc="upper left", ncol=1)
ax_crosstalk.grid(True, **grid_keywords)
ax_crosstalk.set_xlim(left=0)

# ---------------------------------------------------------------------------
# Panel labels  (order: a c e g / b d f)
# ---------------------------------------------------------------------------
_panel_axes = [
    ax_rmse_konrath,   # a
    ax_rmse_hunziker,  # b
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
output_dir  = code_root / "plots" / "paper_plots"
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "real_data_big_figure.pdf"
fig.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"Saved to {output_path}")
# plt.show()

# ---------------------------------------------------------------------------
# Summary statistics: mean rRMSE over ensemble and time
# ---------------------------------------------------------------------------
print("\n--- rMAE summary (mean ± std over ensemble members and time points) ---")
for dataset in ["konrath", "hunziker"]:
    for model in ["ODE", "UDE"]:
        data = load_rmae(RUNS[dataset][model])
        if data is None:
            print(f"{dataset.capitalize()} {model}: no data found")
            continue
        # Average over time for each ensemble member, then summarise across ensemble
        per_member = data.mean(axis=1)          # (n_ensemble,)
        grand_mean = per_member.mean()
        grand_std  = per_member.std()
        print(f"{dataset.capitalize()} {model}: {grand_mean:.4f} ± {grand_std:.4f}  "
              f"(n_ensemble={data.shape[0]}, n_timepoints={data.shape[1]})")
