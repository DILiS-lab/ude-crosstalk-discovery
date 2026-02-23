
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.interpolate import interp1d

# Set up paths
current_file_path = Path(__file__).resolve()
project_root = current_file_path.parents[3]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Experiment Configuration
EXPERIMENT_ROOT = project_root / "experiments/multivariate/multivariate_study_2026-02-10_182131"
MODEL = "hunziker2010"
FUNCTION = "monoton_increasing_weak_hill"
N_SIGNALS = 512
SEEDS = range(10)

def get_experiment_dirs(root_dir):
    """Find all experiment directories matching the criteria."""
    exp_dirs = []
    if not root_dir.exists():
        print(f"Error: Directory {root_dir} not found.")
        return exp_dirs
        
    for item in root_dir.iterdir():
        if item.is_dir():
            # Check if directory name matches the pattern
            name = item.name
            if not name.startswith(MODEL):
                continue
            
            parts = name.split('_')
            
            try:
                # hunziker2010_monoton_increasing_weak_hill_0.01_n128
                
                dir_n_signals_str = parts[-1]
                dir_noise_str = parts[-2]
                
                if dir_n_signals_str != f"n{N_SIGNALS}":
                    continue
                
                # Check Function
                function_name = "_".join(parts[1:-2])
                if function_name != FUNCTION:
                    continue

                # Parse noise
                try:
                    noise_val = float(dir_noise_str)
                except ValueError:
                    continue
                
                exp_dirs.append({
                    "path": item,
                    "noise": noise_val
                })
                
            except Exception as e:
                # print(f"Skipping {name}: {e}")
                continue
                
    return exp_dirs

def load_and_process_data(exp_info):
    """Load data from all seeds and interpolate to a common grid."""
    all_seeds_interp = []
    common_grid = None
    true_function_y = None
    
    # Define a common grid for interpolation based on the range of inputs
    # We'll determine the range from the first valid seed
    grid_min = 0
    grid_max = 1.2 # Default, will update
    resolution = 100
    
    # First, scan seeds to find the data range and the true function
    # (Assuming true function is same for all seeds - which it should be)
    
    collected_data = []

    for seed in SEEDS:
        seed_dir = exp_info["path"] / f"run_seed_{seed}"
        data_file = seed_dir / "learned_vs_true_crosstalk_factor_values.csv"
        
        if not data_file.exists():
            print(f"  Warning: Missing data for seed {seed}")
            continue
            
        try:
            df = pd.read_csv(data_file)
            collected_data.append(df)
        except Exception as e:
            print(f"  Error reading seed {seed}: {e}")

    if not collected_data:
        return None, None, None

    # Determine global min/max for the grid
    all_nfkb = pd.concat([d['nfkb_flat'] for d in collected_data])
    grid_min = all_nfkb.min()
    grid_max = all_nfkb.max()
    
    common_grid = np.linspace(grid_min, grid_max, resolution)
    
    interpolated_preds = []
    
    # Process each seed
    for df in collected_data:
        # Sort by nfkb for interpolation
        df_sorted = df.sort_values(by='nfkb_flat')
        
        # Get True Function (only need to do this once effectively, or average strictly)
        # We'll rely on the first seed's true function, interpolated to the grid
        # (Assuming the "True" function is deterministic and identical)
        if true_function_y is None:
             f_true = interp1d(df_sorted['nfkb_flat'], df_sorted['true_synth_factor'], 
                               kind='linear', fill_value="extrapolate")
             true_function_y = f_true(common_grid)
        
        # Interpolate Predicted Function
        f_pred = interp1d(df_sorted['nfkb_flat'], df_sorted['pred_synth_factor'], 
                          kind='linear', fill_value="extrapolate")
        pred_y = f_pred(common_grid)
        interpolated_preds.append(pred_y)
        
    if not interpolated_preds:
        return None, None, None
        
    return common_grid, true_function_y, np.array(interpolated_preds)

def main():
    print(f"Analyzing {MODEL} with N={N_SIGNALS}, Function={FUNCTION}")
    experiments = get_experiment_dirs(EXPERIMENT_ROOT)
    
    if not experiments:
        print("No matching experiments found.")
        return

    # Sort experiments by noise level
    experiments.sort(key=lambda x: x['noise'])
    
    # Create the plot - Single Figure as requested
    fig, ax = plt.subplots(figsize=(6, 5))
    
    # Defined colors
    colors = ['C0', 'C1', 'C2', 'C3', 'C4']
    
    true_plotted = False
    
    for i, exp in enumerate(experiments):
        noise_val = exp['noise']
        print(f"Processing Noise={noise_val}...")
        
        grid, true_y, preds_array = load_and_process_data(exp)
        
        if grid is None:
            print(f"Skipping Noise={noise_val} - No Data")
            continue
            
        # Calculate statistics
        mean_pred = np.mean(preds_array, axis=0)
        p05_pred = np.percentile(preds_array, 5, axis=0)
        p95_pred = np.percentile(preds_array, 95, axis=0)
        
        # Select Color
        color = colors[i % len(colors)]
        
        # Plot True Crosstalk (Dashed) - Only once
        if not true_plotted and true_y is not None:
            ax.plot(grid, true_y, linestyle='--', color='k', linewidth=2, label='True Crosstalk')
            true_plotted = True
        
        # Plot Learned Mean (Solid)
        ax.plot(grid, mean_pred, linestyle='-', color=color, linewidth=2, label=f'Learned (σ={noise_val})')
        
        # Plot Uncertainty (Shaded)
        ax.fill_between(grid, p05_pred, p95_pred, color=color, alpha=0.15)
        
    ax.set_title(f"Effect of Noise on Discovery\n({MODEL}, N={N_SIGNALS})", fontsize=12)
    ax.set_xlabel("NF-κB Level (a.u.)", fontsize=12)
    ax.set_ylabel("Crosstalk Factor", fontsize=12)
    
    # Legend handling
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    output_filename = f"multivariate_noise_summary_{MODEL}_{FUNCTION}_{N_SIGNALS}.png"
    output_path = EXPERIMENT_ROOT / output_filename
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to {output_path}")
    
    # Also save to current directory
    local_output = project_root / "plots" / "multivariate" / "noise" / output_filename
    local_output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(local_output, dpi=300)
    print(f"Plot also saved to {local_output}")
    plt.show()

if __name__ == "__main__":
    main()
