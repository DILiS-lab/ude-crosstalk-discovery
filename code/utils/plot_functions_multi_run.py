import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re
from jax import random as jrandom
import jax.numpy as jnp

def load_losses(run_dir):
    log_file = Path(run_dir) / "training_log.txt"
    losses = []
    if log_file.exists():
        with open(log_file, "r") as f:
            for line in f:
                # Robust regex for scientific notation (e.g., 1.23e-4, -5.0) and "nan"/"inf"
                match = re.search(r"Average Epoch Loss = ([-\d\.eEnaninf]+)", line, re.IGNORECASE)
                if match:
                    try:
                        val = float(match.group(1))
                        losses.append(val)
                    except ValueError:
                        pass
    return losses

def plot_with_error_bars(ax, x, data, label, color, func_name='mean'):
    """Plot mean/median and IQR shaded area."""
    if func_name == "mean":
        func = np.mean
    elif func_name == "median":
        func = np.median

    mean = func(data, axis=0)
    lower = np.percentile(data, 5, axis=0)
    upper = np.percentile(data, 95, axis=0)
    
    ax.plot(x, mean, label=label, color=color, linewidth=1)
    ax.fill_between(x, lower, upper, color=color, alpha=0.3)
def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_functions_multi_run.py <batch_experiment_directory>")
        sys.exit(1)

    batch_dir = Path(sys.argv[1]).resolve()
    if not batch_dir.exists():
        print(f"Directory not found: {batch_dir}")
        sys.exit(1)

    run_dirs = sorted([d for d in batch_dir.iterdir() if d.is_dir() and (d.name.startswith("run_seed_") or d.name.startswith("run_"))])

    print(f"Found {len(run_dirs)} runs. Loading data...")

    # Load config from the first run or batch folder
    config_file = batch_dir / "config.json"
    if not config_file.exists():
        config_file = run_dirs[0] / "config.json"
    
    with open(config_file, "r") as f:
        config = json.load(f)

    project_root = Path(__file__).resolve().parent
    
    # Load True p53 Values
    true_p53_path = project_root / config["true_p53_values_path"]
    true_p53_values = pd.read_csv(true_p53_path, header=None).values
    
    time_points = np.arange(config["time_start"], config["time_end"] + config["time_step"], config["time_step"])

    all_losses = []
    all_crosstalk_x = None
    all_crosstalk_y = []
    all_rrmse = []
    all_p53_preds = []

    for run_dir in run_dirs:
        # Losses
        losses = load_losses(run_dir)
        if losses:
            all_losses.append(losses)
        
        # Crosstalk
        crosstalk_file = run_dir / "learned_crosstalk_factor_values.csv"
        if crosstalk_file.exists():
            df = pd.read_csv(crosstalk_file)
            if all_crosstalk_x is None:
                all_crosstalk_x = df["nfkb_flat"].values
            all_crosstalk_y.append(df["pred_synth_factor"].values)
        
        # RRMSE
        rrmse_file = run_dir / "rmse_per_timepoint.txt"
        if rrmse_file.exists():
            all_rrmse.append(np.loadtxt(rrmse_file))
        
        # P53 Predictions (assuming shape matches time_points x samples)
        p53_file = run_dir / "p53_generated.csv"
        if p53_file.exists():
            all_p53_preds.append(np.loadtxt(p53_file, delimiter=","))

    # Convert to numpy arrays for easier manipulation
    
    # Handle potentially ragged loss arrays (truncate to min length)
    if all_losses:
        min_epochs = min(len(l) for l in all_losses)
        all_losses = np.array([l[:min_epochs] for l in all_losses])
    else:
        all_losses = np.array([])

    all_crosstalk_y = np.array(all_crosstalk_y)
    all_rrmse = np.array(all_rrmse)
    all_p53_preds = np.array(all_p53_preds) # Shape: (runs, time, samples)

    # Ensure time_points matches the data shape (fix potential floating point arange issues)
    if len(all_p53_preds) > 0:
        data_time_steps = all_p53_preds.shape[1]
        if len(time_points) != data_time_steps:
            print(f"Warning: Re-generated time_points length ({len(time_points)}) mismatches data ({data_time_steps}). Truncating to match.")
            if len(time_points) > data_time_steps:
                 time_points = time_points[:data_time_steps]
            else:
                 # If data is larger, we can't easily extrapolate time, but usually arange produces fewer or equal points?
                 # Actually, usually arange produces MORE if floats are slightly off.
                 pass 

    # 1. Plot Loss Curve
    fig_loss, ax_loss = plt.subplots(figsize=(6, 4))
    epochs = np.arange(1, all_losses.shape[1] + 1)
    plot_with_error_bars(ax_loss, epochs, all_losses, "Training Loss", "C0")
    ax_loss.set_xlabel("Epochs")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Training Loss across runs (Mean & IQR)")
    ax_loss.grid(True, alpha=0.3)
    ax_loss.legend()
    ax_loss.set_ylim(bottom=0, top=1800)
    fig_loss.tight_layout()
    fig_loss.savefig(batch_dir / "batch_loss_curve.png")


    # 2. Plot Predicted Crosstalk Function
    fig_ct, ax_ct = plt.subplots(figsize=(6, 4))
    plot_with_error_bars(ax_ct, all_crosstalk_x, all_crosstalk_y, "Learned Crosstalk", "C0", func_name='mean')
    #for cross_talk in all_crosstalk_y:
    #    ax_ct.plot(all_crosstalk_x, cross_talk, color="k", alpha=0.5, linewidth=1, ls=':')
    ax_ct.set_ylim(bottom=0, top=5)
    ax_ct.set_xlabel("NF-kB values")
    ax_ct.set_ylabel("Crosstalk Factor")
    ax_ct.set_title("Learned Crosstalk Function across runs (Mean & IQR)")
    ax_ct.grid(True, alpha=0.3)
    ax_ct.legend()
    fig_ct.tight_layout()
    fig_ct.savefig(batch_dir / "batch_crosstalk_function.png")

    # 3. Plot Relative RMSE
    fig_rmse, ax_rmse = plt.subplots(figsize=(6, 4))
    plot_with_error_bars(ax_rmse, time_points, all_rrmse, "Relative RMSE", "C0")
    ax_rmse.set_ylim(bottom=0, top=1)
    ax_rmse.set_xlabel("Time [h]")
    ax_rmse.set_ylabel("Relative RMSE")
    ax_rmse.set_title("Relative RMSE per Time Point across runs (Mean & IQR)")
    ax_rmse.grid(True, alpha=0.3)
    ax_rmse.legend()
    fig_rmse.tight_layout()
    fig_rmse.savefig(batch_dir / "batch_relative_rmse.png")

    # 4. True values and predictions of 4 random p53 time series
    # Extract 4 random indices (use fixed seed for consistency in plotting)
    num_samples = true_p53_values.shape[1]
    rng = np.random.default_rng(0)
    random_indices = rng.choice(num_samples, 4, replace=False)
    
    fig_p53, axs = plt.subplots(2, 2, figsize=(6, 5), sharex=True, sharey=True)
    axs = axs.flatten()
    
    for i, idx in enumerate(random_indices):
        ax = axs[i]
        # True values
        ax.plot(time_points, true_p53_values[:, idx], 'k--', label="True", alpha=0.7)
        
        # Predictions across runs for this specific sample idx
        # all_p53_preds has shape (runs, time, samples)
        sample_preds = all_p53_preds[:, :, idx] # (runs, time)
        plot_with_error_bars(ax, time_points, sample_preds, "UDE Prediction", "C0")
        
        ax.set_title(f"Sample {idx}")
        ax.grid(True, alpha=0.3)
        if i >= 2: ax.set_xlabel("Time [h]")
        if i % 2 == 0: ax.set_ylabel("P53 Level")
        if i==1: ax.legend()
        ax.set(ylim=(0, 4000))

    fig_p53.suptitle("P53 Time Series Predictions (Mean & IQR) vs True Values")
    fig_p53.tight_layout()
    fig_p53.savefig(batch_dir / "batch_p53_samples.png")

    print(f"Batch plots saved to {batch_dir}")

if __name__ == "__main__":
    main()
