import numpy as np
import matplotlib.pyplot as plt

import os
import glob
import pandas as pd


def load_rrmse_data(file_path):
    try:
        return np.loadtxt(file_path)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []

def plot_rrmse_function(experiment_dirs, title_prefix=""):

    fig, ax = plt.subplots(figsize=(6, 4))
    
    max_time = 0

    for model_name, folder_path in experiment_dirs.items():
        
        # Adjust path to find where logic is running from
        # Assuming we are in code/utils/plotter, go up to code/ then use path
        search_paths = [
            os.path.join("../../", folder_path),
            folder_path,
            os.path.join("../../../", folder_path)
        ]
        
        full_path = None
        for p in search_paths:
            if os.path.exists(p):
                full_path = p
                break
        
        if not full_path:
            print(f"Could not find path for {model_name}: {folder_path}")
            continue

        # Get Time parameters from config
        config_path = os.path.join(full_path, "config.json")
        if not os.path.exists(config_path):
             # Try first seed folder if main config missing
             seed_conf = glob.glob(os.path.join(full_path, "run_seed_*", "config.json"))
             if seed_conf:
                 config_path = seed_conf[0]

        time_step = 0.1 # Default observed
        time_start = 0.0
        
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    cfg = json.load(f)
                    time_step = cfg.get("time_step", time_step)
                    time_start = cfg.get("time_start", time_start)
            except Exception as e:
                print(f"Warning: Could not read config at {config_path}: {e}")

        all_rrmse = []
        seed_folders = glob.glob(os.path.join(full_path, "run_seed_*"))
        
        for seed_folder in sorted(seed_folders):
            rrmse_file = os.path.join(seed_folder, "rmse_per_timepoint.txt")
            if os.path.exists(rrmse_file):
                rrmse = load_rrmse_data(rrmse_file)
                if len(rrmse) > 0:
                    all_rrmse.append(rrmse)
            
        if not all_rrmse:
            print(f"No RRMSE data found for {model_name}")
            continue

        min_len = min(len(l) for l in all_rrmse)
        all_rrmse_trunc = [l[:min_len] for l in all_rrmse]
        data = np.array(all_rrmse_trunc)
        
        mean_rrmse = np.mean(data, axis=0)
        lower_rrmse = np.percentile(data, 5, axis=0)
        upper_rrmse = np.percentile(data, 95, axis=0)
        
        # Calculate time points in hours
        time_points = np.arange(0.1, 20.1, 0.1)
        
        ax.plot(time_points, mean_rrmse, label=model_name,
                lw=1)
        ax.fill_between(time_points, lower_rrmse, upper_rrmse, alpha=0.3)

    ax.set(xlabel="Time [h]", ylabel="Relative RMSE",
           xlim=(0, 20), ylim=(0, 2))
    ax.legend(loc='best',
              title=f'{title_prefix} Model')
    ax.grid(True, which="both", ls="-", alpha=0.2)

        
    plt.tight_layout()
    plt.show()

    save_dir = 'plots/'
    fig.savefig(f"{save_dir}{title_prefix.lower()}_relative_rmse.png", dpi=200)

ude_dir = {'Hunziker': 'experiments/ude_hunziker2010_uncertainty_2026-01-20_181050',
           'Konrath': 'experiments/ude_konrath2020_uncertainty_2026-01-20_100950'}

ode_dir = {'Hunziker': 'experiments/ode_hunziker2010_uncertainty_2026-01-19_132644',
           'Konrath': 'experiments/ode_konrath2020_uncertainty_2026-01-20_142837'}

print("Plotting UDE Relative RMSE:")
plot_rrmse_function(ude_dir, "UDE")

print("\nPlotting ODE Relative RMSE:")
plot_rrmse_function(ode_dir, "ODE")
