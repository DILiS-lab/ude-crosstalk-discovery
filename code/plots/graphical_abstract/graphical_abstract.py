import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from pathlib import Path
import numpy as np

# ---------------------------------------------------------------------------
# Setup Paths
# ---------------------------------------------------------------------------
project_root = Path(__file__).resolve().parents[3]
code_root = project_root / "code"
plots_dir = code_root / "plots" / "graphical_abstract"
abstract_pdf_path = plots_dir / "graphical abstract_v5.pdf"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Figure size
FIG_SIZE = (7.00787, 3)

# Column Width Ratios for (a), (b/c), (d)
# Determine the widths of the columns here
WIDTH_RATIOS = [1.75, 1.5, 1.5] 

# Indices to plot in panels (b) and (c)
SELECTED_INDICES = [2, 10, 100]
P53_SCALE  = [2, 0.4, 1.05]
NFKB_SCALE = [1.5, 1.05, 0.4]

REAL_ALPHA = 1.0

# ---------------------------------------------------------------------------
# Data Paths
# ---------------------------------------------------------------------------
NFKB_PATH   = code_root / "real_data" / "NCIp65TNFGamma200framesDec2025.csv"
P53_PATH    = code_root / "real_data" / "p53intTNFGamma200framesDec2025.csv"
CROSSTALK_EXP_PATH = code_root / "experiments" / "ude_konrath2020_uncertainty_2026-01-20_100950"
TIME_POINTS = np.arange(0.1, 20.1, 0.1)   # 200 points, 0.1 h spacing

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
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

def _agg_p53(paths, loader):
    arrays = [loader(p) for p in paths if p.exists()]
    arrays = [a for a in arrays if a is not None]
    if not arrays:
        return None
    min_len = min(a.shape[0] for a in arrays)
    return np.stack([a[:min_len] for a in arrays], axis=0)

def load_p53_predictions(exp_dir: Path):
    paths = sorted(exp_dir.glob("run_seed_*/p53_generated.csv"))
    return _agg_p53(paths, lambda p: np.loadtxt(p, delimiter=","))

def plot_band(ax, x, data, color, ls, label, alpha=0.20, lw=1.0):
    mean = data.mean(axis=0)
    lo   = np.percentile(data, 5,  axis=0)
    hi   = np.percentile(data, 95, axis=0)
    ax.plot(x, mean, color=color, ls=ls, lw=lw, label=label)
    ax.fill_between(x, lo, hi, color=color, alpha=alpha)

# ---------------------------------------------------------------------------
# Plot Style
# ---------------------------------------------------------------------------
plt.rcParams.update({"font.size": 6, "axes.labelsize": 6, "axes.titlesize": 6,
                     "xtick.labelsize": 6, "ytick.labelsize": 6,
                     "legend.fontsize": 6, "lines.linewidth": 0.1,
                     "axes.linewidth": 0.5,
                     "xtick.major.width": 0.5, "ytick.major.width": 0.5,
                     "xtick.major.size": 2,   "ytick.major.size": 2})

def create_graphical_abstract():
    # Load Data
    nfkb = np.loadtxt(NFKB_PATH, delimiter=",")
    p53  = np.loadtxt(P53_PATH,  delimiter=",")
    
    # Load Predicted p53
    p53_preds_all = load_p53_predictions(CROSSTALK_EXP_PATH)  # (n_seeds, n_timepoints, n_cells)

    # Create Figure
    fig = plt.figure(figsize=FIG_SIZE)
    
    # Create the main grid: 1 row, 3 columns
    gs_main = gridspec.GridSpec(1, 3, figure=fig, width_ratios=WIDTH_RATIOS, wspace=0.15)
    
    # -----------------------------------------------------------------------
    # Column 1: (a) - Display graphical abstract PNG
    # -----------------------------------------------------------------------
    ax_a = fig.add_subplot(gs_main[0])
    
    import matplotlib.image as mpimg
    
    # Path to the PNG file
    png_path = plots_dir / "4355221f-1.png"
    
    if png_path.exists():
        img = mpimg.imread(str(png_path))
        ax_a.imshow(img)
        ax_a.axis('off')
    else:
        ax_a.text(0.5, 0.5, f"Image not found:\n{png_path.name}", 
                  ha='center', va='center', fontsize=6)
        ax_a.set_xticks([])
        ax_a.set_yticks([])
    
    ax_a.set_title("a", loc="left", fontsize=6, fontweight="bold", pad=2)

    # -----------------------------------------------------------------------
    # Column 2: (b) and (c) - Two rows (Top/Bottom)
    # -----------------------------------------------------------------------
    # Nested GridSpec for the middle column
    gs_col2 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_main[1], hspace=0.2, height_ratios=[2, 1])
    
    # Subplot (b) - Top (p53)
    ax_b = fig.add_subplot(gs_col2[0])
    
    # Plot Data and UDE fits
    if p53_preds_all is not None:
        linestyles = ["-", "--", "dashdot"]
        # Plot each trajectory with scale
        for idx, (i, scale) in enumerate(zip(SELECTED_INDICES, P53_SCALE)):
            ls = linestyles[idx]
            # Plot Data (thin, light) matching scale
            ax_b.plot(TIME_POINTS, p53[:, i] * scale, lw=0.8, alpha=REAL_ALPHA, color="C3", ls=ls)
            
            # Plot UDE (with bands) matching scale
            # p53_preds_all shape: (n_seeds, n_tp, n_cells)
            cell_preds = p53_preds_all[:, :, i] # (n_seeds, n_tp)
            
            # Ensure timepoints match
            n_tp = min(cell_preds.shape[1], len(TIME_POINTS))
            tp = TIME_POINTS[:n_tp]
            cell_preds = cell_preds[:, :n_tp] * scale
            
            color = "C0" 

            plot_band(ax_b, tp, cell_preds, color=color, ls=ls, label="UDE" if idx == 0 else None, alpha=0.2, lw=0.8)
            
    else:
         linestyles = ["-", "--", "dashdot"]
         for idx, (i, scale) in enumerate(zip(SELECTED_INDICES, P53_SCALE)):
            ax_b.plot(TIME_POINTS, p53[:, i] * scale, lw=0.8, alpha=REAL_ALPHA, color="C3", ls=linestyles[idx])

    # Legend for Cell Indices
    legend_elements = [
        Line2D([0], [0], color='C3', lw=0.8, ls='-', label='1'),
        Line2D([0], [0], color='C3', lw=0.8, ls='--', label='2'),
        Line2D([0], [0], color='none', marker='', linestyle='None', label=r'$\vdots$'), 
        Line2D([0], [0], color='C3', lw=0.8, ls='dashdot', label='106')
    ]
    leg = ax_b.legend(handles=legend_elements, title="Cell", loc=(0.725,0.35), frameon=True, ncol=1, title_fontsize=6, handlelength=1.5, handletextpad=0.5, borderaxespad=0.2)
    leg.get_frame().set_linewidth(0.5)
            
    ax_b.set_title("b", loc="left", fontsize=6, fontweight="bold", pad=2)
    ax_b.set_xticks([])
    ax_b.set_yticks([])
    ax_b.set_ylabel(r"p53 + p53$_a$")
    ax_b.set_xlabel("Time")
    # Maintain limits from original plot
    ax_b.set_xlim(0, 20)

    
    # Subplot (c) - Bottom (NF-kB)
    ax_c = fig.add_subplot(gs_col2[1])
    linestyles = ["-", "--", "dashdot"]
    for idx, (i, scale) in enumerate(zip(SELECTED_INDICES, NFKB_SCALE)):
        ax_c.plot(TIME_POINTS, nfkb[:, i] * scale, lw=0.8, alpha=REAL_ALPHA, color="C3", ls=linestyles[idx])

    ax_c.set_title("c", loc="left", fontsize=6, fontweight="bold", pad=2)
    ax_c.set_xticks([])
    ax_c.set_yticks([])
    ax_c.set_ylabel(r"NF-$\kappa$B")
    ax_c.set_xlabel("Time")
    # Maintain limits from original plot
    ax_c.set_xlim(0, 20)

    # -----------------------------------------------------------------------
    # Column 3: (d) - Crosstalk
    # -----------------------------------------------------------------------
    ax_d = fig.add_subplot(gs_main[2])
    
    x_grid, ys = load_crosstalk(CROSSTALK_EXP_PATH)
    if x_grid is not None:
        # Neural Network (Learned)
        plot_band(ax_d, x_grid, ys, color="C2", ls="-", label="Neural Network", lw=0.8, alpha=0.15)
        
        # Symbolic Regression (Ground Truth/Approximation)
        # konrath: [NFkB]^2 + 0.076*[NFkB] + 1.645
        sr_fn = lambda x: (x + 0.076078266) * x + 1.6450808
        ax_d.plot(x_grid, sr_fn(x_grid), color="k", ls="-", lw=0.8, alpha=0.7, 
                  label="\n"+"Symbolic Regression\n" + r"$A + B \frac{[NF\text{-}\kappa B]^n}{K_d^n + [NF\text{-}\kappa B]^n}$")
        
        ax_d.legend(loc="upper left", frameon=False)
        ax_d.set_xlabel(r"NF-$\kappa$B")
        ax_d.set_ylabel(r"p53-NF-$\kappa$B Crosstalk")
    else:
        ax_d.text(0.5, 0.5, "Crosstalk data not found", ha='center', va='center')

    ax_d.set_title("d", loc="left", fontsize=6, fontweight="bold", pad=2)
    ax_d.set_xticks([])
    ax_d.set_yticks([])
    if x_grid is not None:
        ax_d.set_xlim(x_grid.min(), x_grid.max())

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------
    # fig.tight_layout() # Optional, sometimes conflicts with complicated gridspecs
    output_path = plots_dir / "graphical_abstract_structure.pdf"
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved structure to {output_path}")
    # plt.show()

if __name__ == "__main__":
    create_graphical_abstract()
