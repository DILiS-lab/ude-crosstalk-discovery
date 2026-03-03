import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from scipy.interpolate import interp1d

# ---------------------------------------------------------------------------
# Paths & config
# ---------------------------------------------------------------------------
project_root    = Path(__file__).resolve().parents[3]
code_root       = project_root / "code"
EXPERIMENT_ROOT = code_root / "experiments" / "multivariate" / "multivariate_study_2026-02-10_182131"

SEEDS               = range(10)
NOISE_CROSSTALK     = 0.1
N_SIGNALS_CROSSTALK = 128
VALID_N             = [16, 128, 512]
VALID_NOISE         = [0.001, 0.01, 0.1]

FUNCTIONS = [
    ("monoton_decreasing_strong_hill", "Inhibiting Strong"),
    ("monoton_decreasing_weak_hill",   "Inhibiting Weak"),
    ("monoton_increasing_strong_hill", "Activating Strong"),
    ("monoton_increasing_weak_hill",   "Activating Weak"),
]

MODELS = [
    ("konrath2020", "Konrath"),
    ("hunziker2010", "Hunziker"),
]

# ---------------------------------------------------------------------------
# Styling
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

# ---------------------------------------------------------------------------
# Figure layout — 2 rows × 5 cols
#   Col 0      : crosstalk panels (row 0 = Konrath, row 1 = Hunziker)
#   Cols 1–4   : relative MAE panels (same 4 functions × 2 models grid)
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(7.00787, 3), constrained_layout=True)
gs  = fig.add_gridspec(2, 5, width_ratios=[1.2, 1, 1, 1, 1])

# Crosstalk axes — share y (both show crosstalk factor, 0–14)
ax_ct_top = fig.add_subplot(gs[0, 0])
ax_ct_bot = fig.add_subplot(gs[1, 0], sharey=ax_ct_top)
ax_ct     = [ax_ct_top, ax_ct_bot]

# MAE axes — all 8 share x and y with each other
ax_mae = [[None] * 4 for _ in range(2)]
for row_i in range(2):
    for col_i in range(4):
        kw = {} if (row_i == 0 and col_i == 0) else \
             {"sharey": ax_mae[0][0], "sharex": ax_mae[0][0]}
        ax_mae[row_i][col_i] = fig.add_subplot(gs[row_i, col_i + 1], **kw)

ct_colors  = ["C0", "C1", "C2", "C3"]
mae_colors = ["C0", "C1", "C2"]
width      = 0.1
gap        = 1.5
offsets    = [-width * gap, 0, width * gap]
indices    = np.arange(len(VALID_N))

# ---------------------------------------------------------------------------
# Crosstalk panels
# ---------------------------------------------------------------------------
for model_idx, (model_key, model_label) in enumerate(MODELS):
    ax = ax_ct[model_idx]
    experiments = []
    for item in EXPERIMENT_ROOT.iterdir():
        if not item.is_dir() or not item.name.startswith(model_key):
            continue
        parts = item.name.split("_")
        try:
            if int(parts[-1][1:]) != N_SIGNALS_CROSSTALK or float(parts[-2]) != NOISE_CROSSTALK:
                continue
            experiments.append({"path": item, "function": "_".join(parts[1:-2])})
        except Exception:
            continue
    experiments.sort(key=lambda x: x["function"])

    for i, exp in enumerate(experiments):
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
        ax.plot(grid, true_y,             color=color, ls=":", lw=0.5)
        ax.plot(grid, preds.mean(axis=0), color=color, ls="-", lw=0.5)
        ax.fill_between(grid, np.percentile(preds, 5, axis=0),
                        np.percentile(preds, 95, axis=0), color=color, alpha=0.2)

    ax.set_ylabel(f"Crosstalk Factor\n({model_label})", fontsize=6)
    ax.set_ylim(0, 14)
    ax.grid(True, **grid_keywords)

    if model_idx == len(MODELS) - 1:
        ax.set_xlabel(r"NF-$\kappa$B Level (a.u.)", fontsize=6)
    else:
        ax.tick_params(labelbottom=False)

# ---------------------------------------------------------------------------
# Load relative MAE data
# ---------------------------------------------------------------------------
valid_func_keys = {f for f, _ in FUNCTIONS}
rows = []
for item in EXPERIMENT_ROOT.iterdir():
    if not item.is_dir():
        continue
    name  = item.name
    parts = name.split("_")
    model = next((m for m, _ in MODELS if name.startswith(m)), None)
    if model is None:
        continue
    try:
        n_signals = int(parts[-1][1:])
        noise     = float(parts[-2])
        func      = "_".join(parts[1:-2])
    except Exception:
        continue
    if n_signals not in VALID_N or noise not in VALID_NOISE or func not in valid_func_keys:
        continue
    for seed in SEEDS:
        csv = item / f"run_seed_{seed}" / "learned_vs_true_crosstalk_factor_values.csv"
        if not csv.exists():
            continue
        try:
            df        = pd.read_csv(csv).sort_values("nfkb_flat")
            grid      = np.linspace(df["nfkb_flat"].min(), df["nfkb_flat"].max(), 500)
            true_vals = np.interp(grid, df["nfkb_flat"].values, df["true_synth_factor"].values)
            pred_vals = np.interp(grid, df["nfkb_flat"].values, df["pred_synth_factor"].values)
            rel_mae   = np.mean(np.abs(true_vals - pred_vals)) / np.mean(np.abs(true_vals))
            rows.append({"model": model, "function": func,
                         "n_signals": n_signals, "noise": noise, "rel_mae": rel_mae})
        except Exception:
            pass

df_all = pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# Relative MAE panels
# ---------------------------------------------------------------------------
for row_i, (model_key, model_label) in enumerate(MODELS):
    for col_i, (func_key, func_label) in enumerate(FUNCTIONS):
        ax  = ax_mae[row_i][col_i]
        sub = df_all[(df_all["model"] == model_key) & (df_all["function"] == func_key)]

        for i, noise_val in enumerate(VALID_NOISE):
            groups = [
                sub[(sub["n_signals"] == n) & (sub["noise"] == noise_val)]["rel_mae"].values
                for n in VALID_N
            ]
            ax.boxplot(
                groups,
                positions=indices + offsets[i],
                widths=width,
                patch_artist=True,
                showfliers=True,
                boxprops=dict(facecolor=mae_colors[i], color="black", alpha=1, linewidth=0.5),
                medianprops=dict(color="black", linewidth=0.5),
                whiskerprops=dict(color="black", linewidth=0.5),
                capprops=dict(color="black", linewidth=0.5),
                flierprops=dict(marker="o", markerfacecolor="k", markersize=1,
                               linestyle="none", markeredgecolor="k", markeredgewidth=0.5),
            )

        ax.set_xticks(indices)
        ax.set_xticklabels(VALID_N)
        ax.set_yscale("log")
        ax.grid(True, axis="y", which="both", **grid_keywords)
        ax.set_ylim(1e-3, 1e0)

        if row_i == 0:
            ax.set_title(func_label, fontsize=6, pad=3)
            ax.tick_params(labelbottom=False)

        if row_i == len(MODELS) - 1:
            ax.set_xlabel("N", fontsize=6)

        if col_i == 0:
            ax.set_ylabel(f"Relative MAE\n({model_label})", fontsize=6)
        else:
            ax.tick_params(labelleft=False)

# ---------------------------------------------------------------------------
# Panel labels  (a–b: crosstalk; c–j: MAE)
# ---------------------------------------------------------------------------
letters = "abcdefghij"
for i, ax in enumerate(ax_ct):
    ax.text(0.04, 0.97, letters[i], transform=ax.transAxes,
            fontsize=6, fontweight="bold", va="top", ha="left")

for row_i in range(2):
    for col_i in range(4):
        letter = letters[2 + row_i * 4 + col_i]
        ax_mae[row_i][col_i].text(0.04, 0.97, letter,
                                   transform=ax_mae[row_i][col_i].transAxes,
                                   fontsize=6, fontweight="bold", va="top", ha="left")

# ---------------------------------------------------------------------------
# Noise legend — panel c (first MAE panel, top-left), bottom left
# ---------------------------------------------------------------------------
legend_handles = [
    mpatches.Patch(facecolor=mae_colors[i], edgecolor="black", label=str(v), linewidth=0.5)
    for i, v in enumerate(VALID_NOISE)
]
leg = ax_mae[0][0].legend(
    handles=legend_handles,
    title=r"Noise ($\sigma$)",
    loc="lower left",
    ncol=1,
    fontsize=5,
    title_fontsize=5,
    handlelength=1.2,
)
leg.get_frame().set_linewidth(0.5)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
output_dir  = code_root / "plots" / "paper_plots"
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "synth_data_combined_figure.pdf"
fig.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"Saved to {output_path}")
