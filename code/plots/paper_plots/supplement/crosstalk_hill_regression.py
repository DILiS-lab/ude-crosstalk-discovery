from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

SCRIPT_DIR   = Path(__file__).resolve().parent
project_root = Path(__file__).resolve().parents[4]
code_root    = project_root / "code"
exp_root     = code_root / "experiments"

RUNS = {
    "hunziker": exp_root / "ude_hunziker2010_uncertainty_2026-01-20_181050",
    "konrath":  exp_root / "ude_konrath2020_uncertainty_2026-01-20_100950",
}

DATASET_LABELS  = {"hunziker": "Hunziker", "konrath": "Konrath"}
CROSSTALK_COLOR = {"hunziker": "C1", "konrath": "C0"}

# Hill regression best equations from hill_regression_multi_results.txt
HILL_FUNCS = {
    "hunziker": (
        r"$0.778 + 0.263\,\frac{[\mathrm{NF\text{-}\kappa B}]^{3.652}}{0.667^{3.652}+[\mathrm{NF\text{-}\kappa B}]^{3.652}}$",
        lambda x: 0.7779 + 0.2632 * (x**3.6521) / (0.6674**3.6521 + x**3.6521),
    ),
    "konrath": (
        r"$1.668 + 13.830\,\frac{[\mathrm{NF\text{-}\kappa B}]^{2.114}}{3.245^{2.114}+[\mathrm{NF\text{-}\kappa B}]^{2.114}}$",
        lambda x: 1.6676 + 13.8303 * (x**2.1140) / (3.2452**2.1140 + x**2.1140),
    ),
}

grid_keywords = {"ls": "-", "alpha": 0.1, "lw": 0.5, "c": "k"}


def load_crosstalk(exp_dir: Path, n_grid: int = 500):
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


fig, ax = plt.subplots(figsize=(6, 4))

for dataset in ["konrath", "hunziker"]:
    x_grid, ys = load_crosstalk(RUNS[dataset])
    if x_grid is None:
        continue
    color = CROSSTALK_COLOR[dataset]
    label = DATASET_LABELS[dataset]
    plot_band(ax, x_grid, ys, color=color, ls="-", label=f"{label} (NN)")
    hill_label, hill_fn = HILL_FUNCS[dataset]
    ax.plot(x_grid, hill_fn(x_grid), color=color, ls="--", lw=1, label=hill_label)

ax.autoscale(axis="x", tight=True)
ax.grid(True, **grid_keywords)
ax.legend(loc="upper left", ncol=1, fontsize=12)
ax.set_xlabel(r"NF-$\kappa$B")
ax.set_ylabel("Crosstalk factor")
#ax.set_title("d", loc="left", fontweight="bold", pad=2)

plt.tight_layout()
plt.savefig(SCRIPT_DIR / "crosstalk_hill_regression.pdf", dpi=300, bbox_inches="tight")
# plt.show()
