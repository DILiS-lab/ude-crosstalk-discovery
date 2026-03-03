import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from pathlib import Path
from scipy.interpolate import interp1d

# ---------------------------------------------------------------------------
# Paths & config
# ---------------------------------------------------------------------------
project_root    = Path(__file__).resolve().parents[4]
code_root       = project_root / "code"
EXPERIMENT_ROOT = code_root / "experiments" / "multivariate" / "multivariate_study_2026-02-10_182131"

SEEDS               = range(10)
NOISE_CROSSTALK     = 0.1
N_SIGNALS_CROSSTALK = 128
VALID_N             = [16, 128, 512]
VALID_NOISE         = [0.001, 0.01, 0.1]

# ---------------------------------------------------------------------------
# Figure layout
# ---------------------------------------------------------------------------
plt.rcParams.update({"font.size": 6, "axes.labelsize": 6, "axes.titlesize": 6,
                     "xtick.labelsize": 6, "ytick.labelsize": 6,
                     "legend.fontsize": 6, "lines.linewidth": 0.1})

plt.rcParams.update({"font.size": 6, "axes.labelsize": 6, "axes.titlesize": 6,
                     "xtick.labelsize": 6, "ytick.labelsize": 6,
                     "legend.fontsize": 6, "lines.linewidth": 0.1,
                     "axes.linewidth": 0.5,
                     "xtick.major.width": 0.5, "ytick.major.width": 0.5,
                     "xtick.major.size": 2,   "ytick.major.size": 2})

grid_keywords = {"ls": "-", "alpha": 0.1, "lw": 0.1, 'c': 'k'}

fig = plt.figure(figsize=(7.00787, 2), constrained_layout=True)
sf_ct, sf_mae = fig.subfigures(1, 2, width_ratios=[2, 1], wspace=0.01)

# Left subfigure: crosstalk panels side-by-side (1 row × 2 cols), shared y-axis
gs_ct = sf_ct.add_gridspec(nrows=1, ncols=2, wspace=0.01)
ax_ct_konrath  = sf_ct.add_subplot(gs_ct[0, 0])
ax_ct_hunziker = sf_ct.add_subplot(gs_ct[0, 1], sharey=ax_ct_konrath)

# Right subfigure: rMAE boxplots stacked (2 rows × 1 col), shared x-axis
gs_mae = sf_mae.add_gridspec(nrows=2, ncols=1, hspace=0.005)
ax_mae_konrath  = sf_mae.add_subplot(gs_mae[0, 0])
ax_mae_hunziker = sf_mae.add_subplot(gs_mae[1, 0], sharex=ax_mae_konrath, sharey=ax_mae_konrath)
ax_mae_konrath.tick_params(labelbottom=False)

ct_colors  = ["C0", "C1", "C2", "C3"]
mae_colors = ["C0", "C1", "C2"]
width      = 0.1
gap        = 1.5
offsets    = [-width * gap, 0, width * gap]
indices    = np.arange(len(VALID_N))

# ---------------------------------------------------------------------------
# Crosstalk — Konrath  (unchanged from main figure)
# ---------------------------------------------------------------------------
experiments_ct_konrath = []
for item in EXPERIMENT_ROOT.iterdir():
    if not item.is_dir() or not item.name.startswith("konrath2020"):
        continue
    parts = item.name.split("_")
    try:
        if int(parts[-1][1:]) != N_SIGNALS_CROSSTALK or float(parts[-2]) != NOISE_CROSSTALK:
            continue
        experiments_ct_konrath.append({"path": item, "function": "_".join(parts[1:-2])})
    except Exception:
        continue
experiments_ct_konrath.sort(key=lambda x: x["function"])

for i, exp in enumerate(experiments_ct_konrath):
    collected = []
    for seed in SEEDS:
        f = exp["path"] / f"run_seed_{seed}" / "learned_vs_true_crosstalk_factor_values.csv"
        if f.exists():
            try:
                collected.append(pd.read_csv(f))
            except Exception:
                pass
    if not collected:
        continue
    all_x = pd.concat([d["nfkb_flat"] for d in collected])
    grid  = np.linspace(all_x.min(), all_x.max(), 100)
    true_y, pred_list = None, []
    for df in collected:
        df = df.sort_values("nfkb_flat")
        if true_y is None:
            true_y = interp1d(df["nfkb_flat"], df["true_synth_factor"],
                              kind="linear", fill_value="extrapolate")(grid)
        pred_list.append(interp1d(df["nfkb_flat"], df["pred_synth_factor"],
                                  kind="linear", fill_value="extrapolate")(grid))
    preds = np.array(pred_list)
    color = ct_colors[i % len(ct_colors)]
    label = exp["function"].replace("monoton_", "").replace("hill", "").replace("_", " ").strip().title()
    ax_ct_konrath.plot(grid, true_y,             color=color, ls=":", lw=1)
    ax_ct_konrath.plot(grid, preds.mean(axis=0), color=color, ls="-", lw=0.5, label=label)
    ax_ct_konrath.fill_between(grid, np.percentile(preds, 5, axis=0),
                               np.percentile(preds, 95, axis=0), color=color, alpha=0.2)

ax_ct_konrath.set_xlabel(r"NF-$\kappa$B Level (z)")
ax_ct_konrath.grid(True, **grid_keywords)
ax_ct_konrath.set_ylabel(r"Crosstalk Factor $U_\theta(z)$ (Konrath)", fontsize=6)
ax_ct_konrath.set_ylim(0, 14)

# ---------------------------------------------------------------------------
# Crosstalk — Hunziker  (unchanged from main figure)
# ---------------------------------------------------------------------------
experiments_ct_hunziker = []
for item in EXPERIMENT_ROOT.iterdir():
    if not item.is_dir() or not item.name.startswith("hunziker2010"):
        continue
    parts = item.name.split("_")
    try:
        if int(parts[-1][1:]) != N_SIGNALS_CROSSTALK or float(parts[-2]) != NOISE_CROSSTALK:
            continue
        experiments_ct_hunziker.append({"path": item, "function": "_".join(parts[1:-2])})
    except Exception:
        continue
experiments_ct_hunziker.sort(key=lambda x: x["function"])

for i, exp in enumerate(experiments_ct_hunziker):
    collected = []
    for seed in SEEDS:
        f = exp["path"] / f"run_seed_{seed}" / "learned_vs_true_crosstalk_factor_values.csv"
        if f.exists():
            try:
                collected.append(pd.read_csv(f))
            except Exception:
                pass
    if not collected:
        continue
    all_x = pd.concat([d["nfkb_flat"] for d in collected])
    grid  = np.linspace(all_x.min(), all_x.max(), 100)
    true_y, pred_list = None, []
    for df in collected:
        df = df.sort_values("nfkb_flat")
        if true_y is None:
            true_y = interp1d(df["nfkb_flat"], df["true_synth_factor"],
                              kind="linear", fill_value="extrapolate")(grid)
        pred_list.append(interp1d(df["nfkb_flat"], df["pred_synth_factor"],
                                  kind="linear", fill_value="extrapolate")(grid))
    preds = np.array(pred_list)
    color = ct_colors[i % len(ct_colors)]
    label = exp["function"].replace("monoton_", "").replace("hill", "").replace("_", " ").strip().title()
    ax_ct_hunziker.plot(grid, true_y,             color=color, ls=":", lw=1)
    ax_ct_hunziker.plot(grid, preds.mean(axis=0), color=color, ls="-", lw=0.5, label=label)
    ax_ct_hunziker.fill_between(grid, np.percentile(preds, 5, axis=0),
                                np.percentile(preds, 95, axis=0), color=color, alpha=0.2)

ax_ct_hunziker.set_xlabel(r"NF-$\kappa$B Level (z)")
ax_ct_hunziker.grid(True, **grid_keywords)
ax_ct_hunziker.set_ylabel(r"Crosstalk Factor $U_\theta(z)$ (Hunziker)", fontsize=6)
ax_ct_hunziker.set_ylim(0, 14)

# ---------------------------------------------------------------------------
# Relative MAE — Konrath
# Relative MAE per run = MAE / mean(|true_synth_factor|)
# ---------------------------------------------------------------------------
rows_konrath = []
for item in EXPERIMENT_ROOT.iterdir():
    if not item.is_dir() or not item.name.startswith("konrath2020"):
        continue
    parts = item.name.split("_")
    try:
        n_signals = int(parts[-1][1:])
        noise     = float(parts[-2])
    except Exception:
        continue
    for seed in SEEDS:
        f = item / f"run_seed_{seed}" / "learned_vs_true_crosstalk_factor_values.csv"
        if not f.exists():
            continue
        try:
            df        = pd.read_csv(f).sort_values("nfkb_flat")
            grid      = np.linspace(df["nfkb_flat"].min(), df["nfkb_flat"].max(), 500)
            true_vals = np.interp(grid, df["nfkb_flat"].values, df["true_synth_factor"].values)
            pred_vals = np.interp(grid, df["nfkb_flat"].values, df["pred_synth_factor"].values)
            mae       = np.mean(np.abs(true_vals - pred_vals))
            rel_mae   = mae / np.mean(np.abs(true_vals))
            rows_konrath.append({"n_signals": n_signals, "noise": noise, "rel_mae": rel_mae})
        except Exception:
            pass

df_mae_konrath = pd.DataFrame(rows_konrath)
df_mae_konrath = df_mae_konrath[df_mae_konrath["n_signals"].isin(VALID_N) & df_mae_konrath["noise"].isin(VALID_NOISE)]

for i, noise_val in enumerate(VALID_NOISE):
    groups = [df_mae_konrath[(df_mae_konrath["n_signals"] == n) & (df_mae_konrath["noise"] == noise_val)]["rel_mae"].values
              for n in VALID_N]
    ax_mae_konrath.boxplot(groups, positions=indices + offsets[i], widths=width,
                           patch_artist=True, showfliers=True,
                           boxprops=dict(facecolor=mae_colors[i], color="black", alpha=1, linewidth=0.5),
                           medianprops=dict(color="black", linewidth=0.5),
                           whiskerprops=dict(color="black", linewidth=0.5),
                           capprops=dict(color="black", linewidth=0.5),
                           flierprops=dict(marker='o', markerfacecolor='k', markersize=1,
                                          linestyle='none', markeredgecolor='k', markeredgewidth=0.5))

ax_mae_konrath.set_xticks(indices)
ax_mae_konrath.set_xticklabels(VALID_N)
ax_mae_konrath.set_yscale("log")
ax_mae_konrath.set_ylim(1e-3, 1e0)
ax_mae_konrath.set_ylabel("rMAE (Konrath)")
ax_mae_konrath.grid(True, axis="y", which="both", **grid_keywords)
legend_handles = [
    mpatches.Patch(facecolor=mae_colors[i], edgecolor="black", label=str(v), linewidth=0.5)
    for i, v in enumerate(VALID_NOISE)
]
leg = ax_mae_konrath.legend(
    handles=legend_handles,
    title=r"Noise ($\eta$)",
    loc="lower left",
    ncol=3,
    fontsize=6,
    title_fontsize=6,
    handlelength=1.2,
)
leg.get_frame().set_linewidth(0.5)

# ---------------------------------------------------------------------------
# Relative MAE — Hunziker
# ---------------------------------------------------------------------------
rows_hunziker = []
for item in EXPERIMENT_ROOT.iterdir():
    if not item.is_dir() or not item.name.startswith("hunziker2010"):
        continue
    parts = item.name.split("_")
    try:
        n_signals = int(parts[-1][1:])
        noise     = float(parts[-2])
    except Exception:
        continue
    for seed in SEEDS:
        f = item / f"run_seed_{seed}" / "learned_vs_true_crosstalk_factor_values.csv"
        if not f.exists():
            continue
        try:
            df        = pd.read_csv(f).sort_values("nfkb_flat")
            grid      = np.linspace(df["nfkb_flat"].min(), df["nfkb_flat"].max(), 500)
            true_vals = np.interp(grid, df["nfkb_flat"].values, df["true_synth_factor"].values)
            pred_vals = np.interp(grid, df["nfkb_flat"].values, df["pred_synth_factor"].values)
            mae       = np.mean(np.abs(true_vals - pred_vals))
            rel_mae   = mae / np.mean(np.abs(true_vals))
            rows_hunziker.append({"n_signals": n_signals, "noise": noise, "rel_mae": rel_mae})
        except Exception:
            pass

df_mae_hunziker = pd.DataFrame(rows_hunziker)
df_mae_hunziker = df_mae_hunziker[df_mae_hunziker["n_signals"].isin(VALID_N) & df_mae_hunziker["noise"].isin(VALID_NOISE)]

for i, noise_val in enumerate(VALID_NOISE):
    groups = [df_mae_hunziker[(df_mae_hunziker["n_signals"] == n) & (df_mae_hunziker["noise"] == noise_val)]["rel_mae"].values
              for n in VALID_N]
    ax_mae_hunziker.boxplot(groups, positions=indices + offsets[i], widths=width,
                            patch_artist=True, showfliers=True,
                            boxprops=dict(facecolor=mae_colors[i], color="black", alpha=1, linewidth=0.5),
                            medianprops=dict(color="black", linewidth=0.5),
                            whiskerprops=dict(color="black", linewidth=0.5),
                            capprops=dict(color="black", linewidth=0.5),
                            flierprops=dict(marker='o', markerfacecolor='k', markersize=1,
                                           linestyle='none', markeredgecolor='k', markeredgewidth=0.5))

ax_mae_hunziker.set_xticks(indices)
ax_mae_hunziker.set_xticklabels(VALID_N)
ax_mae_hunziker.set_yscale("log")
ax_mae_hunziker.set_ylim(1e-3, 1e0)
ax_mae_hunziker.set_ylabel("rMAE (Hunziker)")
ax_mae_hunziker.grid(True, axis="y", which="both", **grid_keywords)
ax_mae_hunziker.set_xlabel("Number of Time Series (M)", fontsize=6)

# ---------------------------------------------------------------------------
# Legend for panels a & b (line styles only, black)
# ---------------------------------------------------------------------------
style_handles = [
    mlines.Line2D([], [], color="k", ls=":", lw=1, label="True"),
    mlines.Line2D([], [], color="k", ls="-", lw=0.5, label="Predicted"),
]
leg = ax_ct_konrath.legend(handles=style_handles, loc="upper left", fontsize=6,
                           handlelength=1.5, frameon=True, ncols=2)
leg.get_frame().set_linewidth(0.5)

# ---------------------------------------------------------------------------
# Panel labels
# ---------------------------------------------------------------------------
ax_ct_konrath.set_title("a", loc="left", fontsize=6, fontweight="bold", pad=2)
ax_ct_hunziker.set_title("b", loc="left", fontsize=6, fontweight="bold", pad=2)
ax_mae_konrath.set_title("c", loc="left", fontsize=6, fontweight="bold", pad=2)
ax_mae_hunziker.set_title("d", loc="left", fontsize=6, fontweight="bold", pad=2)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
output_dir  = code_root / "plots" / "paper_plots" / "supplement"
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "synth_data_big_figure_rmae.pdf"
fig.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"Saved to {output_path}")
