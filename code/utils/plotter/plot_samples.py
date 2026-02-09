import numpy as np
import matplotlib.pyplot as plt

import os
import glob
import pandas as pd
import json


def load_prediction_data(file_path):
    try:
        return np.loadtxt(file_path, delimiter=",")
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def plot_samples_function(experiment_dirs, title_prefix=""):

    fig, axs = plt.subplots(4, 2, figsize=(6, 8), sharex=True, sharey=False)
    axs = axs.flatten()
    
    true_values = None
    time_points = None
    random_indices = None
    
    # Define colors for different models
    colors = ['C0', 'C1', 'C2', 'C3', 'C4']

    max_vals = {i: 0.0 for i in range(len(axs))}
    
    for i, (model_name, folder_path) in enumerate(experiment_dirs.items()):
        
        # Adjust path to find where logic is running from
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

        # Get Config
        config_path = os.path.join(full_path, "config.json")
        if not os.path.exists(config_path):
             seed_conf = glob.glob(os.path.join(full_path, "run_seed_*", "config.json"))
             if seed_conf:
                 config_path = seed_conf[0]

        time_step = 0.1 
        time_start = 0.0
        current_true_values = None
        
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    cfg = json.load(f)
                    time_step = cfg.get("time_step", time_step)
                    time_start = cfg.get("time_start", time_start)
                    
                    # Try to load True Values if we haven't yet
                    if true_values is None:
                        project_root = os.path.dirname(os.path.dirname(full_path))
                        # Sometimes project root calculation might fail depending on depth
                        # Fallback to searching relative
                        
                        true_p53_path_rel = cfg.get("true_p53_values_path")
                        if true_p53_path_rel:
                            possible_true_paths = [
                                os.path.join(project_root, true_p53_path_rel),
                                os.path.join("../../", true_p53_path_rel),
                                true_p53_path_rel
                            ]
                            for tp in possible_true_paths:
                                if os.path.exists(tp):
                                    try:
                                        current_true_values = pd.read_csv(tp, header=None).values
                                        break
                                    except:
                                        pass
            except Exception as e:
                print(f"Warning: Could not read config at {config_path}: {e}")

        # Load Predictions
        all_preds = []
        seed_folders = glob.glob(os.path.join(full_path, "run_seed_*"))
        
        for seed_folder in sorted(seed_folders):
            pred_file = os.path.join(seed_folder, "p53_generated.csv")
            if os.path.exists(pred_file):
                pred = load_prediction_data(pred_file)
                if pred is not None:
                    all_preds.append(pred)
        
        if not all_preds:
            print(f"No prediction data found for {model_name}")
            continue
            
        # Stack and Handle Truncation
        min_len = min(p.shape[0] for p in all_preds)
        all_preds_trunc = [p[:min_len, :] for p in all_preds]
        data = np.array(all_preds_trunc) # (runs, time, samples)
        
        # Initialize Shared State (True Values, Indices, Time)
        if true_values is None and current_true_values is not None:
            true_values = current_true_values
            
            rng = np.random.default_rng(0) 
            n_samples = true_values.shape[1]
            random_indices = rng.choice(n_samples, 8, replace=False)
            
        # Fallback for indices if no true values found
        if random_indices is None:
             n_samples = data.shape[2]
             rng = np.random.default_rng(0)
             random_indices = rng.choice(n_samples, 8, replace=False)

        # Time points (using current model's step)
        if time_points is None:
             time_points = np.arange(min_len) * time_step + time_start
        else:
            # ensure length matches for plotting
            # if mismatches, we use slicing later
            pass

        random_indices = (24, 47, 10, 35, 51, 90, 78, 2)

        # Plot Predictions
        for ax_idx, sample_idx in enumerate(random_indices):
            ax = axs[ax_idx]
            
            sample_data = data[:, :, sample_idx] # (runs, time)
            
            mean_pred = np.mean(sample_data, axis=0)
            lower_pred = np.percentile(sample_data, 5, axis=0)
            upper_pred = np.percentile(sample_data, 95, axis=0)
            
            # Align time points
            plot_len = min(len(time_points), len(mean_pred))
            
            label = model_name + ' ' + title_prefix
            ax.plot(time_points[:plot_len], mean_pred[:plot_len], label=label, 
                    color=colors[i % len(colors)], lw=1.5)
            ax.fill_between(time_points[:plot_len], lower_pred[:plot_len], upper_pred[:plot_len], 
                            color=colors[i % len(colors)], alpha=0.3)
            ax.set(xlim=(0, 20))
            
            # Update max tracking
            if len(upper_pred) > 0:
                 max_vals[ax_idx] = max(max_vals[ax_idx], np.max(upper_pred[:plot_len]))

    # Plot True Values
    if true_values is not None and random_indices is not None:
        for ax_idx, sample_idx in enumerate(random_indices):
            ax = axs[ax_idx]
            tv = true_values[:, sample_idx]
            
            plot_len = min(len(tv), len(time_points)) if time_points is not None else len(tv)
            t_axis = time_points[:plot_len] if time_points is not None else np.arange(len(tv))
            
            ax.plot(t_axis, tv[:plot_len], 'k--', label="Data", lw=1.5, alpha=0.8)
            ax.text(0.95, 0.1, f"Sample {sample_idx}", transform=ax.transAxes,
                    ha='right', va='center', bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.5))

            if len(tv) > 0:
                 max_vals[ax_idx] = max(max_vals[ax_idx], np.max(tv[:plot_len]))

    # Set Labels and Legend
    for i, ax in enumerate(axs):
        ax.set_ylim(0, max_vals[i] * 1.1)
        ax.grid(True, which="both", ls="-", alpha=0.2)
        if i >= 6:
            ax.set_xlabel("Time [h]")
        if i % 2 == 0:
            ax.set_ylabel("p53 Level")
            
        # Add legend to the second plot (top right)
        if i == 1:
            ax.legend(loc='best', fontsize='small')

    

    plt.tight_layout()
    plt.show()

    save_dir = 'plots/'
    fig.savefig(f"{save_dir}{title_prefix.lower()}_samples.png", dpi=200)

ude_dir = {'Hunziker': 'experiments/ude_hunziker2010_uncertainty_2026-01-20_181050',
           'Konrath': 'experiments/ude_konrath2020_uncertainty_2026-01-20_100950'}

ode_dir = {'Hunziker': 'experiments/ode_hunziker2010_uncertainty_2026-01-19_132644',
           'Konrath': 'experiments/ode_konrath2020_uncertainty_2026-01-20_142837'}

print("Plotting UDE Samples:")
plot_samples_function(ude_dir, "UDE")

print("\nPlotting ODE Samples:")
plot_samples_function(ode_dir, "ODE")
