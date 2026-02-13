
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Set up paths
current_file_path = Path(__file__).resolve()
project_root = current_file_path.parents[3]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Experiment Configuration
EXPERIMENT_ROOT = project_root / "experiments/multivariate/multivariate_study_2026-02-10_182131"
MODEL = "hunziker2010"
SEEDS = range(10)

def get_experiment_dirs(root_dir):
    """Find all experiment directories matching the criteria."""
    exp_dirs = []
    if not root_dir.exists():
        print(f"Error: Directory {root_dir} not found.")
        return exp_dirs
        
    for item in root_dir.iterdir():
        if item.is_dir():
            name = item.name
            if not name.startswith(MODEL):
                continue
            
            parts = name.split('_')
            try:
                dir_n_signals_str = parts[-1]
                dir_noise_str = parts[-2]
                
                n_signals = int(dir_n_signals_str[1:])
                noise = float(dir_noise_str)
                function_name = "_".join(parts[1:-2])
                
                exp_dirs.append({
                    "path": item,
                    "n_signals": n_signals,
                    "noise": noise,
                    "function": function_name
                })
            except Exception:
                continue
    return exp_dirs

def get_seed_metrics(exp_info):
    """Extract metrics for all seeds."""
    seed_metrics = []
    
    for seed in SEEDS:
        seed_dir = exp_info["path"] / f"run_seed_{seed}"
        
        crosstalk_file = seed_dir / "learned_vs_true_crosstalk_factor_values.csv"
        traj_rmse_file = seed_dir / "rmse_overall_p53.txt"
        
        if not crosstalk_file.exists() or not traj_rmse_file.exists():
            continue
            
        try:
            # 1. Calculate Crosstalk MAE
            df_cross = pd.read_csv(crosstalk_file)
            mae_crosstalk = np.mean(np.abs(df_cross['true_synth_factor'] - df_cross['pred_synth_factor']))
            
            # 2. Read Trajectory RMSE
            with open(traj_rmse_file, 'r') as f:
                rmse_traj_str = f.readline().strip()
                rmse_traj = float(rmse_traj_str)
                
            seed_metrics.append({
                "trajectory_rmse": rmse_traj,
                "crosstalk_mae": mae_crosstalk,
                "seed": seed
            })
            
        except Exception:
            pass
            
    return seed_metrics

def main():
    print(f"Generating Identifiability Plot (Fig 3) for {MODEL}...")
    experiments = get_experiment_dirs(EXPERIMENT_ROOT)
    
    if not experiments:
        print("No matching experiments found.")
        return

    all_data = []
    
    print(f"Processing {len(experiments)} experiments...")
    for i, exp in enumerate(experiments):
        metrics = get_seed_metrics(exp)
        for m in metrics:
            all_data.append({
                "n_signals": exp['n_signals'],
                "noise": exp['noise'],
                "function": exp['function'],
                "trajectory_rmse": m['trajectory_rmse'],
                "crosstalk_mae": m['crosstalk_mae']
            })
            
    df = pd.DataFrame(all_data)
    
    if df.empty:
        print("No data collected.")
        return

    # Filter standard setup (optional, but let's plot all valid ones)
    valid_n = [16, 128, 512]
    valid_noise = [0.001, 0.01, 0.1]
    df = df[df['n_signals'].isin(valid_n) & df['noise'].isin(valid_noise)]

    # --- Plotting ---
    fig, axes = plt.subplots(1, 3, figsize=(9, 4), sharey=True, sharex=True)
    
    # Colors for noise levels
    unique_noises = sorted(df['noise'].unique())
    colors = ['C0', 'C1', 'C2'] # Matches previous plots if 3 levels
    
    # Check if we have more noise levels than colors
    if len(unique_noises) > len(colors):
        # Fallback to colormap
        cmap = plt.get_cmap('viridis')
        normalize = plt.Normalize(vmin=min(unique_noises), vmax=max(unique_noises))
        color_map = {n: cmap(normalize(n)) for n in unique_noises}
    else:
        color_map = {n: c for n, c in zip(unique_noises, colors)}
        
    for i, n_val in enumerate(valid_n):
        ax = axes[i]
        df_n = df[df['n_signals'] == n_val]
        
        for noise in unique_noises:
            subset = df_n[df_n['noise'] == noise]
            if subset.empty: 
                continue
            
            ax.scatter(subset['trajectory_rmse'], subset['crosstalk_mae'], 
                       c=color_map[noise], 
                       label=f"σ={noise}",
                       alpha=0.6, 
                       edgecolors='none',
                       s=30)
            
        ax.set_title(f"N = {n_val}")
        ax.set_xlabel("Trajectory Fit (RMSE)")
        if i == 0:
            ax.set_ylabel("Mechanism Recovery (Crosstalk MAE)")
            
        ax.set_aspect('equal', 'box')
        
        # Log scales usually help here as errors span orders of magnitude
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        # Add 1:1 diagonal reference line
        # We need to set limits first or get them, maybe uniform limits across all plots?
        # But autoscaling is probably fine per plot
        
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        
        min_val = min(xlim[0], ylim[0])
        max_val = max(xlim[1], ylim[1])
        
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.3, zorder=0, label="1:1" if i==2 else "")
        
        # Reset limits to what they were (or slightly adjusted for aspect)
        # Using aspect='equal' might force changes. 
        # Ideally we want square plots.
        
        ax.grid(True, which="both", ls="-", alpha=0.2)
        
        if i == 0:
            ax.legend(title="Noise Level", loc='best')
            
    plt.suptitle(f"Identifiability Check: Trajectory Fit vs. Mechanism Discovery ({MODEL})")
    
    output_filename = f"multivariate_identifiability_scatter_{MODEL}_all_N.png"
    local_output = project_root / "plots" / "multivariate" / output_filename
    local_output.parent.mkdir(parents=True, exist_ok=True)
    
    plt.tight_layout()
    plt.savefig(local_output, dpi=300)
    print(f"Plot saved to {local_output}")

if __name__ == "__main__":
    main()
