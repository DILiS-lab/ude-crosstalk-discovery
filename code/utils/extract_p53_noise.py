import numpy as np
import pandas as pd
from scipy.stats import linregress
import os
import glob
import json
import matplotlib.pyplot as plt

multi_run_dir = 'experiments/ude_konrath2020_uncertainty_2026-01-20_100950'

def load_and_flatten(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find file: {path}")

    return pd.read_csv(path, header=None).values.flatten()

def analyze_p53_noise(experiment_dir):
    print(f"--- Analyzing p53 Noise Structure for: {experiment_dir} ---")
    
    config_path = os.path.join(experiment_dir, 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Load Real Data
    real_vals_full = load_and_flatten(config['true_p53_values_path'])

    # Aggregate Predictions from all seeds
    all_pred = []
    all_real = []
    
    seed_folders = sorted(glob.glob(os.path.join(experiment_dir, 'run_seed_*')))
    print(f"Found {len(seed_folders)} runs. Aggregating residuals...")

    for folder in seed_folders:
        pred_vals = load_and_flatten(os.path.join(folder, 'p53_generated.csv'))
        
        # Ensure proper length match
        n = min(len(real_vals_full), len(pred_vals))
        all_real.append(real_vals_full[:n])
        all_pred.append(pred_vals[:n])

    # Combine
    combined_real = np.concatenate(all_real)
    combined_pred = np.concatenate(all_pred)
    residuals = combined_real - combined_pred
    
    # --- Binning Analysis ---
    num_bins = 50
    bins = np.linspace(combined_pred.min(), combined_pred.max(), num_bins + 1)
    
    bin_centers = []
    bin_stds = []
    
    for i in range(num_bins):
        mask = (combined_pred >= bins[i]) & (combined_pred < bins[i+1])
        if np.sum(mask) > 50: 
            bin_centers.append((bins[i] + bins[i+1])/2)
            bin_stds.append(np.std(residuals[mask]))

    bin_centers = np.array(bin_centers)
    bin_stds = np.array(bin_stds)

    # --- Linear Regression ---
    slope, intercept, r_value, _, _ = linregress(bin_centers, bin_stds)

    print(f"\nLinear Regression Results (Noise Sigma vs Signal Intensity):")
    print(f"  Slope (alpha, multiplicative): {slope:.5f}")
    print(f"  Intercept (beta, additive):    {intercept:.5f}")
    print(f"  R-squared:                     {r_value**2:.4f}")
    
    mean_signal = np.mean(bin_centers) if len(bin_centers) > 0 else 0
    contrib_multiplicative = slope * mean_signal
    
    print(f"\nInterpretation at Mean Signal Intensity ({mean_signal:.2f}):")
    print(f"  Multiplicative part (~{slope:.4f} * Signal): {contrib_multiplicative:.2f}")
    print(f"  Additive part (Intercept): {intercept:.2f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

    # 1. Residuals vs Signal (Hexbin) with Model Overlay
    hb = ax1.hexbin(combined_pred, residuals, gridsize=100, mincnt=1, bins='log', cmap='Blues')
    ax1.set_xlabel("Predicted p53")
    ax1.set_ylabel("Residuals (truth - prediction)")
    ax1.set_title("Residuals vs Signal")
    
    # Overlay the fitted sigma model on the residuals
    x_range = np.linspace(combined_pred.min(), combined_pred.max(), 100)
    sigma_model = slope * x_range + intercept
    ax1.plot(x_range, sigma_model, 'k--', label=r'$\pm 1\sigma$ Model')
    ax1.plot(x_range, -sigma_model, 'k--')
    ax1.plot(x_range, 2*sigma_model, 'k:', label=r'$\pm 2\sigma$ Model')
    ax1.plot(x_range, -2*sigma_model, 'k:')
    ax1.legend(loc='upper right')

    # 2. Sigma vs Signal (The Regression)
    ax2.plot(bin_centers, bin_stds, 'ko', label='Binned Std Dev')
    ax2.plot(bin_centers, slope * bin_centers + intercept, 'r-', 
             label=f'Linear Fit\n$\sigma \\approx {slope:.4f}S + {intercept:.1f}$')
    
    ax2.set_xlabel("Predicted p53")
    ax2.set_ylabel("Noise std ($\sigma$)")
    ax2.set_title("Noise Scaling Model")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    fig.savefig('plots/p53_noise_analysis.png', dpi=300)
    
    print(f"\nCONCLUSION:")
    if contrib_multiplicative > 2 * intercept:
        print("  >> The noise is predominantly MULTIPLICATIVE.")
    elif intercept > 2 * contrib_multiplicative:
        print("  >> The noise is predominantly ADDITIVE.")
    else:
        print("  >> The noise is MIXED (both components are significant).")
        
    print(f"\nComputed Noise Parameters (Averaged over {len(seed_folders)} runs):")
    print(f"  sigma_additive       = {intercept:.4f}")
    print(f"  sigma_multiplicative = {slope:.4f}")
    
    return {
        'sigma_additive': intercept,
        'sigma_multiplicative': slope,
        'residuals': residuals,
        'predicted': combined_pred,
        'bin_centers': bin_centers,
        'bin_stds': bin_stds
    }

if __name__ == "__main__":
    analyze_p53_noise(multi_run_dir)
