import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.patches as mpatches

def plot_single_model(model_name, config, experiments_dir, code_dir):
    ude_folder_name = config["ude_folder"]
    ode_folder_name = config["ode_folder"]
    param_names = config["params"]
    
    ude_path = os.path.join(experiments_dir, ude_folder_name)
    ode_path = os.path.join(experiments_dir, ode_folder_name)
    
    # Check if paths exist
    if not os.path.exists(ude_path):
        print(f"Error: UDE path not found: {ude_path}")
        return
    if not os.path.exists(ode_path):
        print(f"Error: ODE path not found: {ode_path}")
        return

    data_frames = []

    def load_data(experiment_path, file_pattern, label):
        # Look for run_seed_* folders
        seed_folders = glob.glob(os.path.join(experiment_path, "run_seed_*"))
        
        dfs = []
        for seed_folder in sorted(seed_folders):
            file_path = os.path.join(seed_folder, file_pattern)
            if os.path.exists(file_path):
                try:
                    df = pd.read_csv(file_path, header=None, names=param_names)
                    df['Type'] = label
                    df['Seed'] = os.path.basename(seed_folder)
                    dfs.append(df)
                    print(f"Loaded {len(df)} rows from {os.path.basename(seed_folder)}/{file_pattern}")
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
            else:
                print(f"Warning: File not found: {file_path}")
        
        if dfs:
            return pd.concat(dfs, ignore_index=True)
        else:
            return pd.DataFrame()

    # 1. Non-fitted (from UDE folder, sampled_parameters.csv)
    print(f"\n--- Processing {model_name} ---")
    print(f"Loading Non-fitted parameters from {ude_folder_name}...")
    df_non_fitted = load_data(ude_path, "sampled_parameters.csv", "Non-fitted")
    if not df_non_fitted.empty:
        data_frames.append(df_non_fitted)

    # 2. Fitted UDE (from UDE folder, learned_model_parameters.csv)
    print(f"Loading Fitted UDE parameters from {ude_folder_name}...")
    df_fitted_ude = load_data(ude_path, "learned_model_parameters.csv", "Fitted UDE")
    if not df_fitted_ude.empty:
        data_frames.append(df_fitted_ude)

    # 3. Fitted non-NFkB (from ODE folder, learned_model_parameters.csv)
    print(f"Loading Fitted non-NFkB parameters from {ode_folder_name}...")
    df_fitted_ode = load_data(ode_path, "learned_model_parameters.csv", "Fitted non-NFkB-activated model")
    if not df_fitted_ode.empty:
        data_frames.append(df_fitted_ode)

    if not data_frames:
        print(f"No data collected for {model_name}.")
        return
        
    full_df = pd.concat(data_frames, ignore_index=True)

    # Plotting
    types = ["Non-fitted", "Fitted UDE", "Fitted non-NFkB-activated model"]
    colors = ['C0', 'C1', 'C2']
    
    # Adjust figure size for Konrath which has many parameters
    figsize = (12, 4) if len(param_names) > 15 else (6, 4)
    
    plt.figure(figsize=figsize, dpi=200)
    ax = plt.gca()

    num_params = len(param_names)
    indices = np.arange(num_params)
    width = 0.2  # Width of box
    offsets = [-width, 0, width]
    
    # Store legend handles
    legend_patches = []

    for i, type_name in enumerate(types):
        # Extract data for this type for all parameters
        data_to_plot = []
        subset = full_df[full_df['Type'] == type_name]
        
        if subset.empty:
            print(f"Warning: No data for {type_name}")
            data_to_plot = [[] for _ in param_names] # Empty placeholders
        else:
            for param in param_names:
                vals = subset[param].dropna().values
                data_to_plot.append(vals)
        
        # Positions for this group
        positions = indices + offsets[i]
        
        # Plot boxplot
        bp = ax.boxplot(data_to_plot, positions=positions, widths=width * 0.8, patch_artist=True,
                        boxprops=dict(facecolor=colors[i]),
                        medianprops=dict(color="black"),
                        flierprops=dict(marker='o', markerfacecolor=colors[i], markersize=2, alpha=0.5))
        
        legend_patches.append(mpatches.Patch(color=colors[i], label=type_name))

    ax.set_xticks(indices)
    ax.set_xticklabels(param_names, rotation=45, ha='right')
    ax.set_title(f"Distribution of Kinetic Parameters ({model_name} Model)")
    ax.set_yscale('log') # Kinetic parameters often vary by orders of magnitude
    ax.grid(True, which="both", ls="-", alpha=0.2)
    ax.legend(handles=legend_patches)
    
    # Save plot
    output_dir = os.path.join(code_dir, "plots")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{model_name.lower()}_kinetic_parameter_distributions.png")
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")
    plt.close()

def plot_kinetic_parameter_distributions():
    # Determine base paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    code_dir = os.path.abspath(os.path.join(current_dir, "../../")) 
    experiments_dir = os.path.join(code_dir, "experiments")
    
    MODELS = {
        "Hunziker": {
            "ude_folder": "ude_hunziker2010_uncertainty_2026-01-20_181050",
            "ode_folder": "ode_hunziker2010_uncertainty_2026-01-19_132644",
            "params": ['sigma_p', 'alpha', 'gamma', 'k_b', 'r', 'k_trs', 'beta', 'k_trd', 'delta']
        },
        "Konrath": {
            "ude_folder": "ude_konrath2020_uncertainty_2026-01-20_100950",
            "ode_folder": "ode_konrath2020_uncertainty_2026-01-20_142837",
            "params": [
                'damage', 'Ts', 'Tw', 'am', 'ampa', 'ampi', 'api', 'as', 'asm', 'aw', 'awpa', 'aws', 
                'bpamt', 'amt', 'bmt', 'bmtm', 'bp', 'bs', 'bsp', 'bpawt', 'awt', 'bwt', 'bwtw', 
                'ns', 'nw', 'bmt_fc', 'bmtm_fc', 'bp_fc', 'bs_fc', 'bwt_fc', 'bwtw_fc'
            ]
        }
    }

    for model_name, config in MODELS.items():
        plot_single_model(model_name, config, experiments_dir, code_dir)

if __name__ == "__main__":
    plot_kinetic_parameter_distributions()
