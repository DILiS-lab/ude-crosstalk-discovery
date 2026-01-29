import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.stats import linregress
import os
import glob
import matplotlib.pyplot as plt

# --- Configuration ---
REAL_DATA_P53 = 'code/real_data/p53intTNFGamma200framesDec2025.csv'
REAL_DATA_NFKB = 'code/real_data/NCIp65TNFGamma200framesDec2025.csv'
EXPERIMENT_ROOT = 'code/experiments/ude_konrath2020_uncertainty_2026-01-20_100950'
ARTIFACTS_DIR = 'code/artifacts'

os.makedirs(ARTIFACTS_DIR, exist_ok=True)

def load_and_flatten(path):
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, header=None)
        return df.values.flatten()
    except Exception as e:
        print(f"Warning: Could not read {path}: {e}")
        return None

def analyze_p53_noise_structure():
    print(f"--- Analyzing p53 Noise Structure (Aggregate of All Runs) ---")
    
    real_vals_full = load_and_flatten(REAL_DATA_P53)
    if real_vals_full is None:
        print("Error: Could not load real p53 data.")
        return

    all_pred = []
    all_real = []
    
    seed_folders = sorted(glob.glob(os.path.join(EXPERIMENT_ROOT, 'run_seed_*')))
    if not seed_folders:
        print(f"Error: No 'run_seed_*' folders found in {EXPERIMENT_ROOT}")
        return

    print(f"Found {len(seed_folders)} runs. Aggregating residuals...")

    for folder in seed_folders:
        pred_path = os.path.join(folder, 'p53_generated.csv')
        pred_vals = load_and_flatten(pred_path)
        
        if pred_vals is not None:
            n = min(len(real_vals_full), len(pred_vals))
            all_real.append(real_vals_full[:n])
            all_pred.append(pred_vals[:n])
            
    if not all_pred:
        print("Error: No prediction data loaded.")
        return

    combined_real = np.concatenate(all_real)
    combined_pred = np.concatenate(all_pred)
    residuals = combined_real - combined_pred
    
    # --- Binning Analysis ---
    num_bins = 30
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
    slope, intercept, r_value, p_value, std_err = linregress(bin_centers, bin_stds)

    print(f"\nLinear Regression Results (Noise Sigma vs Signal Intensity):")
    print(f"  Slope (alpha):       {slope:.5f}")
    print(f"  Intercept (beta):    {intercept:.5f}")
    
    mean_signal = np.mean(bin_centers)
    contrib_multiplicative = slope * mean_signal
    
    print(f"\nInterpretation at Mean Signal Intensity ({mean_signal:.2f}):")
    print(f"  Multiplicative part (~{slope:.4f} * Signal): {contrib_multiplicative:.2f}")
    print(f"  Additive part (Intercept): {intercept:.2f}")
    
    print(f"\nCONCLUSION:")
    if contrib_multiplicative > 2 * intercept:
        print("  >> The noise is predominantly MULTIPLICATIVE.")
    elif intercept > 2 * contrib_multiplicative:
        print("  >> The noise is predominantly ADDITIVE.")
    else:
        print("  >> The noise is MIXED.")
        
    print(f"\nComputed Noise Parameters (Averaged over {len(seed_folders)} runs):")
    print(f"  sigma_additive       = {intercept:.4f}")
    print(f"  sigma_multiplicative = {slope:.4f}")

    # --- Plotting ---
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    if len(combined_pred) > 10000:
        idx = np.random.choice(len(combined_pred), 10000, replace=False)
        plt.scatter(combined_pred[idx], residuals[idx], alpha=0.1, s=1, color='blue')
    else:
        plt.scatter(combined_pred, residuals, alpha=0.2, s=2, color='blue')
        
    plt.axhline(0, color='k', linestyle='--', alpha=0.5)
    plt.xlabel("Predicted p53 Signal Magnitude")
    plt.ylabel("Residual (True - Predicted)")
    plt.title(f"Residuals vs Signal (N={len(seed_folders)} runs)")
    
    plt.subplot(1, 2, 2)
    plt.plot(bin_centers, bin_stds, 'ro-', label='Binned Noise Std')
    plt.plot(bin_centers, slope*bin_centers + intercept, 'k--', 
             label=f'Fit: {slope:.3f}x + {intercept:.1f}')
    
    plt.xlabel("Predicted p53 Signal Magnitude")
    plt.ylabel("Measured Noise Std Deviation")
    plt.title("Noise Structure Analysis")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_path = os.path.join(ARTIFACTS_DIR, 'p53_noise_analysis_multirun.png')
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"\nSaved p53 noise analysis plot to: {output_path}")

def estimate_nfkb_noise():
    print(f"\n--- Estimating NF-kB Input Noise ---")
    if not os.path.exists(REAL_DATA_NFKB):
        print(f"Error: {REAL_DATA_NFKB} not found.")
        return

    df = pd.read_csv(REAL_DATA_NFKB, header=None)
    data = df.values 
    
    if data.shape[0] < data.shape[1]:
        data = data.T
        
    noise_stds = []
    
    for i in range(data.shape[1]):
        trajectory = data[:, i]
        trajectory = trajectory[~np.isnan(trajectory)]
        if len(trajectory) < 15: continue
            
        try:
            smoothed = savgol_filter(trajectory, window_length=11, polyorder=3)
            resi = trajectory - smoothed
            noise_stds.append(np.std(resi))
        except:
            pass
        
    avg_noise_std = np.mean(noise_stds)
    print(f"Estimated NF-kB Noise Std Dev (sigma): {avg_noise_std:.4f}")

    # --- Plotting ---
    plt.figure(figsize=(10, 6))
    
    n_examples = min(3, data.shape[1])
    for i in range(n_examples):
        traj = data[:, i]
        traj = traj[~np.isnan(traj)]
        if len(traj) < 15: continue
            
        smoothed = savgol_filter(traj, window_length=11, polyorder=3)
        
        offset = i * (np.max(data[~np.isnan(data)]) * 0.5)
        
        plt.plot(traj + offset, '.', label=f'Raw Traj {i}', alpha=0.5)
        plt.plot(smoothed + offset, '-', label=f'Smoothed Traj {i}')
        
    plt.title(f"NF-kB Noise Extraction (Sigma ~ {avg_noise_std:.4f})")
    plt.xlabel("Time")
    plt.ylabel("NF-kB Intensity (Offset for clarity)")
    plt.legend()
    
    output_path = os.path.join(ARTIFACTS_DIR, 'nfkb_noise_estimation.png')
    plt.savefig(output_path)
    print(f"Saved NF-kB noise estimation plot to: {output_path}")

if __name__ == "__main__":
    analyze_p53_noise_structure()
    estimate_nfkb_noise()
