import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths & config
# ---------------------------------------------------------------------------
project_root    = Path(__file__).resolve().parents[4]
code_root       = project_root / "code"
EXPERIMENT_ROOT = code_root / "experiments" / "multivariate" / "multivariate_study_2026-02-10_182131"

SEEDS       = range(10)
VALID_N     = [16, 128, 512]
VALID_NOISE = [0.001, 0.01, 0.1]

# 4 crosstalk functions (columns) — ordered for display
FUNCTIONS = [
    ("monoton_decreasing_strong_hill", "Decr. Strong"),
    ("monoton_decreasing_weak_hill",   "Decr. Weak"),
    ("monoton_increasing_strong_hill", "Incr. Strong"),
    ("monoton_increasing_weak_hill",   "Incr. Weak"),
]

# 2 p53 UDE models (rows)
MODELS = [
    ("konrath2020", "Konrath"),
    ("hunziker2010", "Hunziker"),
]

# ---------------------------------------------------------------------------
# Load all MAE data
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
    if n_signals not in VALID_N or noise not in VALID_NOISE:
        continue
    if func not in valid_func_keys:
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
            mae       = np.mean(np.abs(true_vals - pred_vals))
            rows.append({"model": model, "function": func,
                         "n_signals": n_signals, "noise": noise, "mae": mae})
        except Exception:
            pass

df_all = pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# Figure layout — 2 rows × 4 cols, shared axes
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

fig, axes = plt.subplots(
    2, 4,
    figsize=(7.00787, 3.5),
    sharex=True, sharey=True,
    constrained_layout=True,
)

mae_colors = ["C0", "C1", "C2"]
width      = 0.1
gap        = 1.5
offsets    = [-width * gap, 0, width * gap]
indices    = np.arange(len(VALID_N))
panel_letters = "abcdefgh"

for row_i, (model_key, model_label) in enumerate(MODELS):
    for col_i, (func_key, func_label) in enumerate(FUNCTIONS):
        ax  = axes[row_i, col_i]
        sub = df_all[(df_all["model"] == model_key) & (df_all["function"] == func_key)]

        for i, noise_val in enumerate(VALID_NOISE):
            groups = [
                sub[(sub["n_signals"] == n) & (sub["noise"] == noise_val)]["mae"].values
                for n in VALID_N
            ]
            ax.boxplot(
                groups,
                positions=indices + offsets[i],
                widths=width,
                patch_artist=True,
                showfliers=True,
                boxprops=dict(facecolor=mae_colors[i], color="black", alpha=1),
                medianprops=dict(color="black"),
                whiskerprops=dict(color="black"),
                capprops=dict(color="black"),
                flierprops=dict(marker="o", markerfacecolor="k", markersize=1,
                               linestyle="none", markeredgecolor="k"),
            )

        ax.set_xticks(indices)
        ax.set_xticklabels(VALID_N)
        ax.set_yscale("log")
        ax.grid(True, axis="y", which="both", **grid_keywords)

        # Panel letter (top-left, inside panel)
        letter = panel_letters[row_i * 4 + col_i]
        ax.text(0.04, 0.97, letter, transform=ax.transAxes,
                fontsize=6, fontweight="bold", va="top", ha="left")

        # Column header: function name on top row
        if row_i == 0:
            ax.set_title(func_label, fontsize=6, fontweight="bold", pad=3)

        # x-axis label on bottom row only
        if row_i == len(MODELS) - 1:
            ax.set_xlabel("Number of Time Series (N)", fontsize=6)

        # y-axis label on leftmost column only
        if col_i == 0:
            ax.set_ylabel(f"MAE ({model_label})", fontsize=6)

# ---------------------------------------------------------------------------
# Shared noise-level legend (below all panels)
# ---------------------------------------------------------------------------
legend_handles = [
    mpatches.Patch(facecolor=mae_colors[i], edgecolor="black", label=str(v))
    for i, v in enumerate(VALID_NOISE)
]
fig.legend(
    handles=legend_handles,
    title=r"Noise ($\sigma$)",
    loc="lower center",
    ncol=len(VALID_NOISE),
    fontsize=5,
    title_fontsize=5,
    columnspacing=1,
    handlelength=1.2,
    bbox_to_anchor=(0.5, -0.04),
)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
output_dir  = code_root / "plots" / "paper_plots" / "supplement"
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "mae_supplement_figure.pdf"
fig.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"Saved to {output_path}")
