
import os
import glob
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import sympy
from pysr import PySRRegressor

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
    
    print("Configuring PySRRegressor...")

    # PySR Configuration based on user prompt
    
    model = PySRRegressor(
        niterations=50,
        binary_operators=["+", "*", "-", "/", "pow"],
        unary_operators=[],
        constraints={"pow": (1, 1)},
        complexity_of_operators={"/": 1, "+": 1, "*": 1, "-": 1, "pow": 1},
        nested_constraints={"pow": {"+": 0, "-": 0, "*": 0, "/": 0, "pow": 0}},
        parsimony=10, # Increased parsimony to favor simpler models
        
        maxsize=8, # Restrict size to prevent overfitting with high-degree polynomials
        model_selection="best",
        temp_equation_file=True,
        verbosity=1,
        progress=True,
        random_state=0,
        deterministic=True,
        parallelism='serial'
    )
    
    print("Fitting symbolic regression model...")
    model.fit(X, y, weights=weights)
    
    best_equation = model.get_best()
    print("\n" + "="*50)
    print("Best equation found:", best_equation["equation"])
    print("="*50 + "\n")
    
    # Save results
    output_log = os.path.join(experiment_dir, "symbolic_new_regression_multi_results.txt")
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
    plt.plot(x_grid, y_sym, c='C3', ls='--', linewidth=1, label=f"Symbolic Fit\n{best_equation['equation']}")
    
    plt.xlabel("NF-kB values")
    plt.ylabel("Crosstalk Factor")
    plt.title("Symbolic Regression on Learned Crosstalk")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(bottom=0, top=5)
    plt.tight_layout()
    
    plot_file = os.path.join(experiment_dir, "symbolic_new_regression_multi_plot.png")
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
