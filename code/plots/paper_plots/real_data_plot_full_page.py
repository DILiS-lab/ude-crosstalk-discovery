import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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

grid_keywords = {"ls": "-", "alpha": 0.1, "lw": 0.1, 'c': 'k'}

# ---------------------------------------------------------------------------
# Load data  –  shape (200 time points, N cells)
# ---------------------------------------------------------------------------
nfkb = np.loadtxt(NFKB_PATH, delimiter=",")
p53  = np.loadtxt(P53_PATH,  delimiter=",")

# ---------------------------------------------------------------------------
# Figure layout  –  full page width, 3 panels in a row
# ---------------------------------------------------------------------------
plt.rcParams.update({"font.size": 6, "axes.labelsize": 6, "axes.titlesize": 6,
                     "xtick.labelsize": 6, "ytick.labelsize": 6,
                     "legend.fontsize": 6, "lines.linewidth": 0.1,
                     "axes.linewidth": 0.5,
                     "xtick.major.width": 0.5, "ytick.major.width": 0.5,
                     "xtick.major.size": 2,   "ytick.major.size": 2})

fig = plt.figure(figsize=(7.00787, 1.5))

gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.25, width_ratios=[3, 3, 2])

ax_p53       = fig.add_subplot(gs[0, 0])
ax_nfkb      = fig.add_subplot(gs[0, 1])
ax_nfkb_dist = fig.add_subplot(gs[0, 2])

ax_p53.set_title("a", loc="left", fontsize=6, fontweight="bold", pad=2)
ax_nfkb.set_title("b", loc="left", fontsize=6, fontweight="bold", pad=2)
ax_nfkb_dist.set_title("c", loc="left", fontsize=6, fontweight="bold", pad=2)

# ---------------------------------------------------------------------------
# a  –  p53 time series
# ---------------------------------------------------------------------------
for i in range(p53.shape[1]):
    ax_p53.plot(TIME_POINTS, p53[:, i], lw=0.75, alpha=0.5)
#ax_p53.plot(TIME_POINTS, p53.mean(axis=1), color='k', lw=0.5, label="Mean")
ax_p53.set_xlim(0, 20)
ax_p53.set_ylim(0, 12000)
ax_p53.set_xlabel("Time [h]")
ax_p53.set_ylabel("p53", labelpad=2.5)
ax_p53.grid(**grid_keywords)

# ---------------------------------------------------------------------------
# b  –  NF-κB time series
# ---------------------------------------------------------------------------
for i in range(nfkb.shape[1]):
    ax_nfkb.plot(TIME_POINTS, nfkb[:, i], lw=0.75, alpha=0.5)
#ax_nfkb.plot(TIME_POINTS, nfkb.mean(axis=1), color='k', lw=0.5, label="Mean")
ax_nfkb.set_xlim(0, 20)
ax_nfkb.set_ylim(0, 1.75)
ax_nfkb.set_xlabel("Time [h]")
ax_nfkb.set_ylabel(r"NF-$\kappa$B", labelpad=2.5)
ax_nfkb.grid(**grid_keywords)

# ---------------------------------------------------------------------------
# c  –  NF-κB distribution (log scale)
# ---------------------------------------------------------------------------
ax_nfkb_dist.hist(nfkb.ravel(), bins=50,
                  color='k', alpha=0.75, density=False, histtype='stepfilled')
ax_nfkb_dist.set_xlabel(r"NF-$\kappa$B")
ax_nfkb_dist.set_ylabel("Count", labelpad=2.5)
ax_nfkb_dist.set_xlim(0, 1.75)
ax_nfkb_dist.set_yscale("log")
ax_nfkb_dist.grid(**grid_keywords)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
fig.tight_layout()
output_dir  = code_root / "plots" / "paper_plots"
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "real_data_plot_full_page.pdf"
fig.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"Saved to {output_path}")
# plt.show()
