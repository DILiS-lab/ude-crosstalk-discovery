
import os
import glob
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import sympy
from scipy.optimize import curve_fit

def hill_increasing(x, offset, scale, Kd, n):
    return offset + scale * (x**n) / (Kd**n + x**n)

def hill_decreasing(x, offset, scale, Kd, n):
    return offset + scale * (Kd**n) / (Kd**n + x**n)

def load_and_aggregate_data(experiment_dir, n_points=1000):
    """
    Loads crosstalk data from all runs, interpolates to a common grid,
    and calculates the ensemble mean.
    """
    # Find all run directories
    run_dirs = glob.glob(os.path.join(experiment_dir, "run_seed_*"))
    if not run_dirs:
        raise ValueError(f"No run_seed_* directories found in {experiment_dir}")
    
    valid_runs = []
    min_x = float('inf')
    max_x = float('-inf')
    
    print(f"Found {len(run_dirs)} runs. Loading data...")
    
    for run_dir in run_dirs:
        file_path = os.path.join(run_dir, "learned_crosstalk_factor_values.csv")
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} not found. Skipping.")
            continue
            
        df = pd.read_csv(file_path)
        required_cols = ["nfkb_flat", "pred_synth_factor"]
        if not all(col in df.columns for col in required_cols):
            print(f"Warning: {file_path} missing columns {required_cols}. Skipping.")
            continue
            
        # Ensure sorted by x
        df = df.sort_values(by="nfkb_flat")
        
        # Track global range (union of ranges)
        min_x = min(min_x, df["nfkb_flat"].min())
        max_x = max(max_x, df["nfkb_flat"].max())
        
        valid_runs.append(df)
    
    start_x = df["nfkb_flat"].min()
    end_x = df["nfkb_flat"].max()

    print(f"Interpolating on range [{start_x}, {end_x}] with {n_points} points.")
    
    x_grid = np.linspace(start_x, end_x, n_points)
    interpolated_ys = []
    
    for df in valid_runs:
        # Interpolate
        # Uses constant extrapolation if needed (though we try to stay in range)
        # But np.interp uses constant extrapolation by default? No, it uses the edge values.
        y_interp = np.interp(x_grid, df["nfkb_flat"].values, df["pred_synth_factor"].values)
        interpolated_ys.append(y_interp)
        
    interpolated_ys = np.array(interpolated_ys)
    
    y_mean = np.mean(interpolated_ys, axis=0)
    y_std = np.std(interpolated_ys, axis=0)
    
    return x_grid, y_mean, y_std, interpolated_ys

def run_symbolic_regression_multi(experiment_dir):
    print(f"Processing experiment: {experiment_dir}")
    
    x_grid, y_mean, y_std, all_ys = load_and_aggregate_data(experiment_dir)

    # Reshape for PySR (2D arrays required for X)
    X = x_grid.reshape(-1, 1)
    y = y_mean
    
    # Calculate weights - Inverse Variance Weighting
    # w = 1 / var
    weights = 1.0 / y_std**2
    sigma = y_std.copy()
    sigma[sigma < 1e-9] = 1e-9 # Prevent division by zero/singularities in curve_fit

    print("Fitting constrained Hill models (Increasing/Decreasing) using curve_fit instead of PySR...")
    
    # Initial guesses: offset, scale, Kd, n
    # offset ~ min(y)
    # scale ~ max(y) - min(y)
    # Kd ~ mean(x)
    # n ~ 2.0
    p0 = [np.min(y), np.max(y) - np.min(y), np.mean(x_grid), 2.0]
    
    # Bounds:
    # offset: 0 to inf
    # scale: 0 to inf
    # Kd: > 0.001
    # n: > 0.1
    bounds = ([0.0, 0.0, 1e-6, 0.1], [np.inf, np.inf, np.max(x_grid)*2, 10.0])
    
    results = []

    # 1. Increasing Fit
    try:
        popt1, _ = curve_fit(hill_increasing, x_grid, y, p0=p0, sigma=sigma, absolute_sigma=True, bounds=bounds, maxfev=50000)
        y_pred1 = hill_increasing(x_grid, *popt1)
        mse1 = np.average((y - y_pred1)**2, weights=weights)
        eq1 = f"{popt1[0]:.4f} + {popt1[1]:.4f} * (x**{popt1[3]:.4f}) / ({popt1[2]:.4f}**{popt1[3]:.4f} + x**{popt1[3]:.4f})"
        latex1 = rf"${popt1[0]:.3f} + \frac{{ {popt1[1]:.3f} \cdot x^{{{popt1[3]:.3f}}} }}{{ {popt1[2]:.3f}^{{{popt1[3]:.3f}}} + x^{{{popt1[3]:.3f}}} }}$"
        results.append({'popt': popt1, 'mse': mse1, 'eq': eq1, 'latex': latex1, 'func': hill_increasing, 'name': 'Increasing'})
    except Exception as e:
        print(f"Increasing model fit failed: {e}")

    # 2. Decreasing Fit
    try:
        popt2, _ = curve_fit(hill_decreasing, x_grid, y, p0=p0, sigma=sigma, absolute_sigma=True, bounds=bounds, maxfev=50000)
        y_pred2 = hill_decreasing(x_grid, *popt2)
        mse2 = np.average((y - y_pred2)**2, weights=weights)
        eq2 = f"{popt2[0]:.4f} + {popt2[1]:.4f} * ({popt2[2]:.4f}**{popt2[3]:.4f}) / ({popt2[2]:.4f}**{popt2[3]:.4f} + x**{popt2[3]:.4f})"
        latex2 = rf"${popt2[0]:.3f} + \frac{{ {popt2[1]:.3f} \cdot {popt2[2]:.3f}^{{{popt2[3]:.3f}}} }}{{ {popt2[2]:.3f}^{{{popt2[3]:.3f}}} + x^{{{popt2[3]:.3f}}} }}$"
        results.append({'popt': popt2, 'mse': mse2, 'eq': eq2, 'latex': latex2, 'func': hill_decreasing, 'name': 'Decreasing'})
    except Exception as e:
        print(f"Decreasing model fit failed: {e}")
        
    if not results:
        print("Error: Both model fits failed.")
        return

    # Select best model (lowest weighted MSE)
    best_res = min(results, key=lambda x: x['mse'])
    
    print("\n" + "="*50)
    print(f"Best equation found ({best_res['name']}):", best_res["eq"])
    print("="*50 + "\n")
    
    # Mock PySR model interface for compatibility with rest of the script
    class MockModel:
        def __init__(self, best_result, all_results):
            self.best_result = best_result
            self.equations_ = pd.DataFrame([
                {'equation': r['eq'], 'mse': r['mse'], 'model': r['name']} for r in all_results
            ])
            
        def get_best(self):
            return {"equation": self.best_result["eq"]}
            
        def predict(self, X):
            # X comes in as (N, 1) or (N,)
            x_in = X.flatten() if hasattr(X, 'flatten') else X
            return self.best_result['func'](x_in, *self.best_result['popt'])
            
        def score(self, X, y):
            # Return negative MSE or R2? Existing code just prints score. 
            # PySR score is usually negative MSE or custom.
            return -self.best_result['mse']

    model = MockModel(best_res, results)
    best_equation = model.get_best()
    
    # Save results
    output_log = os.path.join(experiment_dir, "hill_regression_multi_results.txt")
    with open(output_log, "w") as f:
        f.write(f"Experiment: {experiment_dir}\n")
        f.write(f"Best equation: {best_equation['equation']}\n")
        f.write(f"Model score: {model.score(X, y)}\n")
        f.write("\nPySR Model Equations:\n")
        
        # Try to write the dataframe of equations
        try:
            f.write(model.equations_.to_string())
        except:
            pass
            
    print(f"Results saved to {output_log}")

    # Plotting
    print("Generating plot...")
    plt.figure(figsize=(6, 4))
    
    # Plot spread (5-95% Range)
    q05 = np.percentile(all_ys, 5, axis=0)
    q95 = np.percentile(all_ys, 95, axis=0)
        
    plt.fill_between(x_grid, q05, q95, color='C0', alpha=0.3, label="Learned Crosstalk (5-95% Range)")
    
    # Plot mean
    plt.plot(x_grid, y_mean, color='C0', linewidth=1, label="Learned Crosstalk (Ensemble Mean)")
    
    # Plot symbolic fit
    y_sym = model.predict(X)
    plt.plot(x_grid, y_sym, c='C3', ls='--', linewidth=1, label=f"Symbolic Fit\n{best_res['latex']}")
    
    plt.xlabel("NF-kB values")
    plt.ylabel("Crosstalk Factor")
    plt.title("Symbolic Regression on Learned Crosstalk")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(bottom=0, top=5)
    plt.tight_layout()
    
    plot_file = os.path.join(experiment_dir, "hill_regression_multi_plot.png")
    plt.savefig(plot_file, dpi=300)
    print(f"Plot saved to {plot_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run symbolic regression on aggregated UDE crosstalk results.")
    parser.add_argument("experiment_dir", type=str, help="Path to the experiment directory containing run_seed_* subdirectories")
    args = parser.parse_args()
    
    if os.path.isdir(args.experiment_dir):
        run_symbolic_regression_multi(args.experiment_dir)
    else:
        print(f"Error: Directory '{args.experiment_dir}' not found.")
