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
            
            min_x = df["nfkb_flat"].min()
            max_x = df["nfkb_flat"].max()
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue
            
    if not valid_runs:
        return None, None, None

    # Use common range for interpolation
    x_grid = np.linspace(min_x, max_x, n_points)
    interpolated_ys = []
    
    for df in valid_runs:
        y_interp = np.interp(x_grid, df["nfkb_flat"].values, df["pred_synth_factor"].values)
        interpolated_ys.append(y_interp)
        
    all_ys = np.array(interpolated_ys)
    return x_grid, all_ys

def load_sr_equation(experiment_dir):
    results_file = os.path.join(experiment_dir, "symbolic_regression_multi_results.txt")
    if not os.path.exists(results_file):
        print(f"SR results file not found: {results_file}")
        return None
        
    equation = None
    try:
        with open(results_file, 'r') as f:
            for line in f:
                if line.startswith("Best equation:"):
                    equation = line.split("Best equation:")[1].strip()
                    break
    except Exception as e:
        print(f"Error reading SR results: {e}")
        
    return equation

def evaluate_equation(equation_str, x_values):
    """
    Evaluates a string equation (from PySR) on a numpy array x_values.
    Assumes equation uses 'x0' variable.
    """
    if not equation_str:
        return None
        
    # Prepare environment for eval
    # Map x0 to our x_values
    local_env = {"x0": x_values, "np": np}
    
    # Handle common math functions if they appear in PySR output
    # Usually PySR outputs python-compatible syntax, but might need mapping
    # e.g., 'sin' -> 'np.sin' is not automatic in eval unless in locals
    
    # Simple heuristic replacement for common functions (add more if needed)
    math_funcs = ["sin", "cos", "exp", "log", "sqrt", "abs"]
    
    eq_eval = equation_str
    
    # Replace variable if needed (PySR uses x0, x1...)
    # We assume 1D input, so x0 is the only one.
    
    try:
        # Use simple eval with numpy namespace
        # We need to make sure 'sin' is seen as 'np.sin'
        for func in math_funcs:
            # Look for whole word match to avoid replacing substrings incorrectly
            pattern = fr"\b{func}\b"
            eq_eval = re.sub(pattern, f"np.{func}", eq_eval)
            
        y = eval(eq_eval, {"__builtins__": None}, local_env)
        
        # If result is a scaler (constant equation), broadcast to x shape
        if np.isscalar(y):
            y = np.full_like(x_values, y)
            
        return y
    except Exception as e:
        print(f"Error evaluating equation '{equation_str}': {e}")
        return None


def plot_crosstalk_function(experiment_dirs, title_prefix=""):

    fig, ax = plt.subplots(figsize=(6, 4))
    
    colors = ['C0', 'C1', 'C2', 'C3'] # C0, C1 for learned; C2, C3 for SR? 
    # Or maybe color by Model (Hunziker vs Konrath) and use style for Type (Learned vs SR)
    
    # Strategy: 
    # Hunziker: Blue (Learned), Blue Dashed (SR)
    # Konrath: Orange (Learned), Orange Dashed (SR)
    
    colors_dict = {
        'Hunziker': 'C0',
        'Konrath': 'C1'
    }

    for model_name, folder_path in experiment_dirs.items():
        
        # Path resolution logic similar to previous scripts
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
        
        if x_grid is None:
            print(f"No crosstalk data found for {model_name}")
        else:
            y_mean = np.mean(all_ys, axis=0)
            y_05 = np.percentile(all_ys, 5, axis=0)
            y_95 = np.percentile(all_ys, 95, axis=0)
            
            c = colors_dict.get(model_name, 'k')
            
            ax.plot(x_grid, y_mean, label=f"{model_name} (NN)", color=c, lw=1)
            ax.fill_between(x_grid, y_05, y_95, color=c, alpha=0.2)
            
        # 2. Load SR Equation
        eq_str = load_sr_equation(full_path)
        if eq_str and x_grid is not None:
            y_sr = evaluate_equation(eq_str, x_grid)
            if y_sr is not None:
                c = colors_dict.get(model_name, 'k')
                
                if model_name == 'Hunziker':
                    label_text = f"$[NF\kappa B] \cdot 0.274 + 0.717$"
                elif model_name == 'Konrath':
                    label_text = f"$[NF\kappa B]^2 + [NF\kappa B] \cdot 0.0761 + 1.645$"
                else:
                    label_text = f"${eq_str}$"
                
                ax.plot(x_grid, y_sr, label=label_text, color=c, ls='--', lw=2)
                print(f"  SR Equation: {eq_str}")
        
    ax.set(xlabel=r"[NF-$\kappa$B]", ylabel="Crosstalk Factor",
           xlim=(x_grid.min(), x_grid.max()), ylim=(0, 4)) # Adjusted limits based on typical NFkB values and crosstalk mag
    ax.legend(loc='upper left')
    ax.grid(True, which="both", ls="-", alpha=0.2)

    plt.tight_layout()
    plt.show()

    save_dir = 'plots/'
    fig.savefig(f"{save_dir}{title_prefix.lower()}_crosstalk.png", dpi=200)

ude_dir = {'Hunziker': 'experiments/ude_hunziker2010_uncertainty_2026-01-20_181050',
           'Konrath': 'experiments/ude_konrath2020_uncertainty_2026-01-20_100950'}

print("Plotting Crosstalk Functions:")
plot_crosstalk_function(ude_dir, "Discovered")
