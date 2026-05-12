"""Print mean MAE for sigma=0.01 experiments across all N settings."""
import numpy as np
import pandas as pd
from pathlib import Path

project_root    = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = project_root / "code" / "experiments" / "multivariate" / "multivariate_study_2026-02-10_182131"

SEEDS     = range(10)
NOISE_VAL = 0.01
VALID_N   = [16, 128, 512]


def collect_mae(prefix):
    rows = []
    for item in EXPERIMENT_ROOT.iterdir():
        if not item.is_dir() or not item.name.startswith(prefix):
            continue
        parts = item.name.split("_")
        try:
            n_signals = int(parts[-1][1:])
            noise     = float(parts[-2])
        except Exception:
            continue
        if noise != NOISE_VAL or n_signals not in VALID_N:
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
                rows.append({"n_signals": n_signals, "mae": np.mean(np.abs(true_vals - pred_vals))})
            except Exception:
                pass
    return pd.DataFrame(rows)


for model, prefix in [("Konrath", "konrath2020"), ("Hunziker", "hunziker2010")]:
    df = collect_mae(prefix)
    print(f"\n{model} (sigma=0.01):")
    for n in VALID_N:
        vals = df[df["n_signals"] == n]["mae"].values
        print(f"  N={n:>3d}  mean={vals.mean():.4f}  median={np.median(vals):.4f}")
