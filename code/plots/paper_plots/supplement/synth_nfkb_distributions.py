"""
Supplement figure: NF-κB distributions for synthetic experiments at M = 16, 128, 512.

Seed handling rationale
-----------------------
`nfkb_generated.csv` is stored at the *experiment* level (one directory per
model × crosstalk_function × noise × M combination), not inside any per-seed
sub-folder.  A spot-check confirms that every experiment sharing the same M
value contains byte-identical NF-κB data (np.allclose → True), because signal
generation is completely independent of model choice, crosstalk function, noise
level, and training seed.  Seeds only affect the UDE training runs that follow.

Consequently, seeds are irrelevant for this plot.  We load one representative
`nfkb_generated.csv` per M value (the first experiment directory that matches
each M) and plot its distribution alongside the real NF-κB data for reference.
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import numpy as np

# ---------------------------------------------------------------------------
# Paths & config
# ---------------------------------------------------------------------------
project_root    = Path(__file__).resolve().parents[4]
code_root       = project_root / "code"
EXPERIMENT_ROOT = code_root / "experiments" / "multivariate" / "multivariate_study_2026-02-10_182131"
NFKB_REAL_PATH  = code_root / "real_data" / "NCIp65TNFGamma200framesDec2025.csv"

M_VALUES = [16, 128, 512]
BINS     = 100

# ---------------------------------------------------------------------------
# Load real NF-κB data (reference)
# ---------------------------------------------------------------------------
nfkb_real = np.loadtxt(NFKB_REAL_PATH, delimiter=",")  # shape (200, N_cells)

# ---------------------------------------------------------------------------
# Load one representative nfkb_generated.csv per M value
# (all experiments with the same M share identical NF-κB data)
# ---------------------------------------------------------------------------
nfkb_synth = {}
for M in M_VALUES:
    suffix = f"_n{M}"
    for item in sorted(EXPERIMENT_ROOT.iterdir()):
        if item.is_dir() and item.name.endswith(suffix):
            csv_path = item / "nfkb_generated.csv"
            if csv_path.exists():
                nfkb_synth[M] = np.loadtxt(csv_path, delimiter=",")
                break
    if M not in nfkb_synth:
        raise FileNotFoundError(f"Could not find nfkb_generated.csv for M={M}")

# ---------------------------------------------------------------------------
# Figure layout  –  1 row × 4 panels (real + M=16 + M=128 + M=512)
# ---------------------------------------------------------------------------
plt.rcParams.update({"font.size": 6, "axes.labelsize": 6, "axes.titlesize": 6,
                     "xtick.labelsize": 6, "ytick.labelsize": 6,
                     "legend.fontsize": 6, "lines.linewidth": 0.1,
                     "axes.linewidth": 0.5,
                     "xtick.major.width": 0.5, "ytick.major.width": 0.5,
                     "xtick.major.size": 2,   "ytick.major.size": 2})

grid_keywords = {"ls": "-", "alpha": 0.1, "lw": 0.1, "c": "k"}

# Determine shared x-axis range across real + synthetic
all_vals  = np.concatenate([nfkb_real.ravel()] + [nfkb_synth[M].ravel() for M in M_VALUES])
x_min, x_max = 0.0, all_vals.max() * 1.05
bin_edges = np.linspace(x_min, x_max, BINS + 1)

fig = plt.figure(figsize=(7.00787, 1.5))  # 1 column = 3.5 inches, so 2 columns = 7 inches
gs  = gridspec.GridSpec(1, 4, figure=fig, wspace=0.35)

axes   = [fig.add_subplot(gs[0, j]) for j in range(4)]
titles = ["a", "b", "c", "d"]
labels = [rf"Real ($N={nfkb_real.shape[1]}$ cells)",
          rf"Synthetic ($M=16$)",
          rf"Synthetic ($M=128$)",
          rf"Synthetic ($M=512$)"]
datasets = [nfkb_real] + [nfkb_synth[M] for M in M_VALUES]

for ax, title, label, data in zip(axes, titles, labels, datasets):
    ax.hist(data.ravel(), bins=bin_edges,
            color="C0", alpha=0.75, density=False,
            histtype="stepfilled")
    ax.set_title(title, loc="left", fontsize=6, fontweight="bold", pad=2)
    ax.set_xlabel(r"NF-$\kappa$B", labelpad=2)
    ax.set_xlim(x_min, x_max)
    ax.set_yscale("log")
    ax.set_ylim(bottom=1e0, top=1e4)
    ax.grid(**grid_keywords)
    ax.text(0.97, 0.97, label, transform=ax.transAxes,
            ha="right", va="top", fontsize=5)

axes[0].set_ylabel("Count", labelpad=2.5)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
fig.tight_layout()
output_dir  = code_root / "plots" / "paper_plots" / "supplement"
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "synth_nfkb_distributions.pdf"
fig.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"Saved to {output_path}")
# plt.show()
