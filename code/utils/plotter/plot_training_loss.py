import numpy as np
import matplotlib.pyplot as plt

import os
import glob
import pandas as pd


def parse_training_log(file_path):
    losses = []
    with open(file_path, 'r') as f:
        for line in f:
            if "Average Epoch Loss =" in line:
                try:
                    loss_part = line.split("Average Epoch Loss =")[1].strip()
                    losses.append(float(loss_part))
                except ValueError:
                    continue
    return losses

def plot_loss_function(experiment_dirs, title_prefix=""):

    fig, ax = plt.subplots(figsize=(6, 4))
    
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

        all_losses = []
        seed_folders = glob.glob(os.path.join(full_path, "run_seed_*"))
        
        for seed_folder in sorted(seed_folders):
            log_file = os.path.join(seed_folder, "training_log.txt")
            if os.path.exists(log_file):
                losses = parse_training_log(log_file)
                all_losses.append(losses)
            
        min_len = min(len(l) for l in all_losses)
        all_losses_trunc = [l[:min_len] for l in all_losses]
        data = np.array(all_losses_trunc)
        
        mean_loss = np.mean(data, axis=0)
        lower_loss = np.percentile(data, 5, axis=0)
        upper_loss = np.percentile(data, 95, axis=0)
        epochs = np.arange(1, min_len + 1)
        
        ax.plot(epochs, mean_loss, label=model_name,
                lw=1)
        ax.fill_between(epochs, lower_loss, upper_loss, alpha=0.3)

        ax.set(xlim=(100,3000), ylim=(100, 2000),
               xlabel="Epoch", ylabel="Training Loss",
               xscale='log', yscale='log',
               xticks=(1, 10, 100, 1000),
               yticks=(100, 1000))
        ax.legend(loc='lower left',
                  title=f'{title_prefix} Model')
        ax.grid(True, which="both", ls="-", alpha=0.2)

        
    plt.tight_layout()
    plt.show()

    save_dir = 'plots/'
    fig.savefig(f"{save_dir}{title_prefix.lower()}_training_loss.png", dpi=200)

ude_dir = {'Hunziker': 'experiments/ude_hunziker2010_uncertainty_2026-01-20_181050',
           'Konrath': 'experiments/ude_konrath2020_uncertainty_2026-01-20_100950'}

ode_dir = {'Hunziker': 'experiments/ode_hunziker2010_uncertainty_2026-01-19_132644',
           'Konrath': 'experiments/ode_konrath2020_uncertainty_2026-01-20_142837'}

print("Plotting UDE Loss:")
plot_loss_function(ude_dir, "UDE")

print("\nPlotting ODE Loss:")
plot_loss_function(ode_dir, "ODE")