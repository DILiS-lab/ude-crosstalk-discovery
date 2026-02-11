import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import pandas as pd
import re

def load_crosstalk_data(experiment_dir, n_points=1000):
    run_dirs = glob.glob(os.path.join(experiment_dir, "run_seed_*"))
    valid_runs = []
    
    min_x = float('inf')
    max_x = float('-inf')

    for run_dir in run_dirs:
        file_path = os.path.join(run_dir, "learned_crosstalk_factor_values.csv")
        if not os.path.exists(file_path):
            continue
            
        try:
            df = pd.read_csv(file_path)
            if "nfkb_flat" not in df.columns or "pred_synth_factor" not in df.columns:
                continue
            
            df = df.sort_values(by="nfkb_flat")
            valid_runs.append(df)
            
            # Update min/max to cover all ranges
            current_min = df["nfkb_flat"].min()
            current_max = df["nfkb_flat"].max()
            if current_min < min_x: min_x = current_min
            if current_max > max_x: max_x = current_max
            
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue
            
    if not valid_runs:
        return None, None, None

    if min_x == float('inf'):
        return None, None, None

    # Use common range for interpolation
    x_grid = np.linspace(min_x, max_x, n_points)
    interpolated_ys = []
    
    for df in valid_runs:
        y_interp = np.interp(x_grid, df["nfkb_flat"].values, df["pred_synth_factor"].values)
        interpolated_ys.append(y_interp)
        
    all_ys = np.array(interpolated_ys)
    return x_grid, all_ys

def load_hill_equation(experiment_dir):
    results_file = os.path.join(experiment_dir, "hill_regression_multi_results.txt")
    if not os.path.exists(results_file):
        print(f"Hill results file not found: {results_file}")
        return None
        
    equation = None
    try:
        with open(results_file, 'r') as f:
            for line in f:
                if line.startswith("Best equation:"):
                    equation = line.split("Best equation:")[1].strip()
                    break
    except Exception as e:
        print(f"Error reading Hill results: {e}")
        
    return equation

def evaluate_equation(equation_str, x_values):
    """
    Evaluates a string equation on a numpy array x_values.
    Assumes equation uses 'x' variable.
    """
    if not equation_str:
        return None
        
    # Prepare environment for eval
    local_env = {"x": x_values, "np": np}
    
    math_funcs = ["sin", "cos", "exp", "log", "sqrt", "abs"]
    
    eq_eval = equation_str
    
    try:
        for func in math_funcs:
            pattern = fr"\b{func}\b"
            eq_eval = re.sub(pattern, f"np.{func}", eq_eval)
            
        y = eval(eq_eval, {"__builtins__": None}, local_env)
        
        if np.isscalar(y):
            y = np.full_like(x_values, y)
            
        return y
    except Exception as e:
        print(f"Error evaluating equation '{equation_str}': {e}")
        return None


def plot_hill_crosstalk_function(experiment_dirs, title_prefix="", show_hill_eq=True):

    fig, ax = plt.subplots(figsize=(6, 4))
    
    colors_dict = {
        'Hunziker': 'C0',
        'Konrath': 'C1'
    }

    x_min_all = float('inf')
    x_max_all = float('-inf')

    for model_name, folder_path in experiment_dirs.items():
        
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

        print(f"Processing {model_name}...")

        # 1. Load Learned Crosstalk
        x_grid, all_ys = load_crosstalk_data(full_path)
        
        c = colors_dict.get(model_name, 'k')
        
        if x_grid is None:
            print(f"No crosstalk data found for {model_name}")
        else:
            if x_grid.min() < x_min_all: x_min_all = x_grid.min()
            if x_grid.max() > x_max_all: x_max_all = x_grid.max()

            y_mean = np.mean(all_ys, axis=0)
            y_05 = np.percentile(all_ys, 5, axis=0)
            y_95 = np.percentile(all_ys, 95, axis=0)
            
            ax.plot(x_grid, y_mean, label=f"{model_name} (NN)", color=c, lw=1)
            ax.fill_between(x_grid, y_05, y_95, color=c, alpha=0.2)
            
        # 2. Load Hill Equation
        if show_hill_eq:
            eq_str = load_hill_equation(full_path)
            if eq_str and x_grid is not None:
                y_hill = evaluate_equation(eq_str, x_grid)
                if y_hill is not None:
                    # Use a simple label
                    if model_name == 'Hunziker':
                        label_text = r'$\frac{[NF\kappa B]^{3.652}}{0.067^{3.652} + [NF\kappa B]^{3.652}} \times 0.263 + 0.078$'
                    if model_name == 'Konrath':
                        label_text = r'$\frac{[NF\kappa B]^{2.114}}{3.245^{2.114} + [NF\kappa B]^{2.114}} \times 13.830 + 1.668$'
                    
                    ax.plot(x_grid, y_hill, label=label_text, color=c, ls='--', lw=2)
                    print(f"  Hill Equation: {eq_str}")
        
    ax.set(xlabel=r"[NF-$\kappa$B]", ylabel="Crosstalk Factor",
           xlim=(x_min_all if x_min_all != float('inf') else 0, x_max_all if x_max_all != float('-inf') else 1), 
           ylim=(0, 4)) 
    ax.legend(loc='upper left')
    ax.grid(True, which="both", ls="-", alpha=0.2)
    
    plt.tight_layout()
    # plt.show()

    save_dir = 'plots/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    fig.savefig(f"{save_dir}{title_prefix.lower()}_hill_crosstalk.png", dpi=200)
    print(f"\nSaved plot to {save_dir}{title_prefix.lower()}_hill_crosstalk.png")

ude_dir = {'Hunziker': 'experiments/ude_hunziker2010_uncertainty_2026-01-20_181050',
           'Konrath': 'experiments/ude_konrath2020_uncertainty_2026-01-20_100950'}

if __name__ == "__main__":
    print("Plotting Hill Crosstalk Functions:")
    print("\nPlotting NN Only Crosstalk Functions:")
    plot_hill_crosstalk_function(ude_dir, "Discovered_NN_Only", show_hill_eq=False)
    plot_hill_crosstalk_function(ude_dir, "Discovered")
