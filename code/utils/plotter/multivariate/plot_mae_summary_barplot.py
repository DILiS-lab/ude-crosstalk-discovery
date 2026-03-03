
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
MODELS = [("hunziker2010", "hunziker"), ("konrath2020", "konrath")]
SEEDS = range(10)

def get_experiment_dirs(root_dir, model):
    """Find all experiment directories matching the criteria."""
    exp_dirs = []
    if not root_dir.exists():
        print(f"Error: Directory {root_dir} not found.")
        return exp_dirs

    for item in root_dir.iterdir():
        if item.is_dir():
            name = item.name
            if not name.startswith(model):
                continue
            
            parts = name.split('_')
            
            try:
                # hunziker2010_monoton_increasing_weak_hill_0.01_n128
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
                
            except Exception as e:
                # print(f"Skipping {name}: {e}")
                continue
                
    return exp_dirs

def compute_mae(exp_info):
    """Compute MAE for all seeds in an experiment."""
    maes = []
    
    for seed in SEEDS:
        seed_dir = exp_info["path"] / f"run_seed_{seed}"
        data_file = seed_dir / "learned_vs_true_crosstalk_factor_values.csv"
        
        if not data_file.exists():
            continue
            
        try:
            df = pd.read_csv(data_file)
            mae = np.mean(np.abs(df['true_synth_factor'] - df['pred_synth_factor']))
            maes.append(mae)
            
        except Exception as e:
            # print(f"  Error reading seed {seed}: {e}")
            pass
            
    return maes

def main():
    for MODEL, subdir in MODELS:
        print(f"Analyzing {MODEL}...")
        experiments = get_experiment_dirs(EXPERIMENT_ROOT, MODEL)

        if not experiments:
            print("No matching experiments found.")
            continue

        # Collect data
        all_results = []

        total_exps = len(experiments)
        print(f"Found {total_exps} experiment folders. Processing...")

        for i, exp in enumerate(experiments):
            if i % 10 == 0:
                print(f"  Processed {i}/{total_exps}...")

            maes = compute_mae(exp)
            for mae in maes:
                all_results.append({
                    "n_signals": exp['n_signals'],
                    "noise": exp['noise'],
                    "function": exp['function'],
                    "mae": mae
                })

        df = pd.DataFrame(all_results)

        if df.empty:
            print("No data collected.")
            continue

        # Filter for the requested values (just in case)
        valid_n = [16, 128, 512]
        valid_noise = [0.001, 0.01, 0.1]

        df = df[df['n_signals'].isin(valid_n) & df['noise'].isin(valid_noise)]

        # Plotting
        plt.figure(figsize=(6, 4))

        # Ensure categorical ordering
        df['n_signals'] = pd.Categorical(df['n_signals'], categories=valid_n, ordered=True)
        df['noise'] = pd.Categorical(df['noise'], categories=valid_noise, ordered=True)

        # Defined colors
        colors = ['C0', 'C1', 'C2']

        # Matplotlib boxplot implementation
        width = 0.15
        gap = 1.25
        offsets = [-width*gap, 0, gap*width] # Positions relative to group center
        indices = np.arange(len(valid_n))

        ax = plt.gca()

        for i, noise_val in enumerate(valid_noise):
            # Collect data for this noise level across all N groups
            data_groups = []
            for n_val in valid_n:
                subset = df[(df['n_signals'] == n_val) & (df['noise'] == noise_val)]['mae'].values
                data_groups.append(subset)

            # Boxplot for this noise level
            positions = indices + offsets[i]
            bp = ax.boxplot(data_groups, positions=positions, widths=width,
                            patch_artist=True, showfliers=True,
                            boxprops=dict(facecolor=colors[i], color='black', alpha=1),
                            medianprops=dict(color='black'),
                            whiskerprops=dict(color='black'),
                            capprops=dict(color='black'))

        # Formatting
        ax.set_xticks(indices)
        ax.set_xticklabels(valid_n)

        plt.title(f"Crosstalk Discovery Error across all Functions ({MODEL})")
        plt.xlabel("Number of Time Series (N)")
        plt.ylabel("Mean Absolute Error (MAE)")
        plt.yscale('log')
        plt.grid(True, axis='y', alpha=0.3, which='both')

        # Custom Legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=colors[i], edgecolor='black', alpha=1, label=str(v))
                           for i, v in enumerate(valid_noise)]
        ax.legend(handles=legend_elements, title="Noise Level (σ)")

        output_filename = f"multivariate_mae_summary_boxplot_{MODEL}.png"

        # Save to model-specific subdirectory
        local_output = project_root / "plots" / "multivariate" / subdir / output_filename
        local_output.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(local_output, dpi=300)
        print(f"Plot saved to {local_output}")
        plt.close()

if __name__ == "__main__":
    main()
