import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

# ---------------------------------------------------------------------------
# Paths & config
# ---------------------------------------------------------------------------
project_root    = Path(__file__).resolve().parents[3]
code_root       = project_root / "code"

NFKB_PATH   = code_root / "real_data" / "NCIp65TNFGamma200framesDec2025.csv"
P53_PATH    = code_root / "real_data" / "p53intTNFGamma200framesDec2025.csv"
TIME_POINTS = np.arange(0.1, 20.1, 0.1)   # 200 points, 0.1 h spacing

NFKB_COLOR = "C0"
P53_COLOR  = "C1"

grid_keywords = {"ls": "-", "alpha": 0.1, "lw": 0.1, 'c': 'k'}

# ---------------------------------------------------------------------------
# Load data  –  shape (200 time points, N cells)
# ---------------------------------------------------------------------------
nfkb = np.loadtxt(NFKB_PATH, delimiter=",")
p53  = np.loadtxt(P53_PATH,  delimiter=",")

# ---------------------------------------------------------------------------
# Figure layout
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

import matplotlib.gridspec as gridspec

# ... (previous code) ...

fig = plt.figure(figsize=(3.38583, 3))

# Create a 2x2 grid with specific spacing
# hspace: height between rows; wspace: width between columns
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.35)

# Top row: two separate plots
ax_nfkb      = fig.add_subplot(gs[0, 0])
ax_nfkb_dist = fig.add_subplot(gs[0, 1])

# Bottom row: one plot spanning both columns
ax_p53       = fig.add_subplot(gs[1, :])

ax_nfkb.set_title("a", loc="left", fontsize=6, fontweight="bold", pad=2)
ax_nfkb_dist.set_title("b", loc="left", fontsize=6, fontweight="bold", pad=2)
ax_p53.set_title("c", loc="left", fontsize=6, fontweight="bold", pad=2)

# ---------------------------------------------------------------------------
# NF-κB time series  (top-left)
# ---------------------------------------------------------------------------
for i in range(nfkb.shape[1]):
    ax_nfkb.plot(TIME_POINTS, nfkb[:, i], lw=0.5, alpha=0.25)

ax_nfkb.plot(TIME_POINTS, nfkb.mean(axis=1),
             color='k', lw=0.5, label="Mean")
ax_nfkb.set_xlim(0, 20)
ax_nfkb.set_ylim(bottom=0)
ax_nfkb.set_ylabel(r"NF-$\kappa$B")
ax_nfkb.set_xlabel("Time [h]")
ax_nfkb.set_ylim(0, 1.75)
ax_nfkb.grid(**grid_keywords)

# ---------------------------------------------------------------------------
# NF-κB distribution  (top-right)
# ---------------------------------------------------------------------------
ax_nfkb_dist.hist(nfkb.ravel(), bins=100,
                  color='k', alpha=1, density=False,  histtype='stepfilled')
ax_nfkb_dist.set_xlabel(r"NF-$\kappa$B")
ax_nfkb_dist.set_ylabel("Count")
ax_nfkb_dist.set_xlim(0, 1.75)
ax_nfkb_dist.grid(**grid_keywords)
# ---------------------------------------------------------------------------
# p53 time series  (bottom, full width)
# ---------------------------------------------------------------------------
for i in range(p53.shape[1]):
    ax_p53.plot(TIME_POINTS, p53[:, i], lw=0.5, alpha=0.25)
ax_p53.plot(TIME_POINTS, p53.mean(axis=1),
            color='k', lw=0.5, label="Mean")
ax_p53.set_xlim(0, 20)
ax_p53.set_ylim(bottom=0)
ax_p53.set_xlabel("Time [h]")
ax_p53.set_ylabel("p53")
ax_p53.grid(**grid_keywords)
ax_p53.set_ylim(0, 12000)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
fig.tight_layout()
output_dir  = code_root / "plots" / "paper_plots"
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "real_data_plot.pdf"
fig.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"Saved to {output_path}")
# plt.show()
