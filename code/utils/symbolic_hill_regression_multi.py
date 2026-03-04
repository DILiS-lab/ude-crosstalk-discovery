
import os
import re
import glob
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from pysr import PySRRegressor, TemplateExpressionSpec


def parse_hill_params(equation_str):
    """
    Parse clean Hill parameters from PySR TemplateExpression equation string.
    Format: "f = #1 * c; p = [p1, p2, p3, p4]"

    f(x) = c * x  →  (c*x)^n / (K + (c*x)^n) = x^n / (K/c^n + x^n)
    So the true Kd = (K / c^n)^(1/n).

    Returns dict with offset, scale, n, Kd, and form ("increasing"/"decreasing").
    """
    # Extract c from f = #1 * c  or  f = c * #1  or  f = #1 (c=1)
    c = 1.0
    m = re.search(r'#1\s*\*\s*([0-9.eE+\-]+)', equation_str)
    if m:
        c = float(m.group(1))
    else:
        m = re.search(r'([0-9.eE+\-]+)\s*\*\s*#1', equation_str)
        if m:
            c = float(m.group(1))

    # Extract p = [p1, p2, p3, p4]
    m = re.search(r'p\s*=\s*\[([^\]]+)\]', equation_str)
    if not m:
        return None
    p = [float(v.strip()) for v in m.group(1).split(',')]
    offset, scale, K, n = p[0], p[1], p[2], p[3]

    # Fold c into K to get the canonical Kd
    Kd = (K / c**n) ** (1.0 / n)
    form = "increasing" if scale > 0 else "decreasing"

    return {"offset": offset, "scale": scale, "K": K, "n": n, "c": c, "Kd": Kd, "form": form}

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

        df = df.sort_values(by="nfkb_flat")

        min_x = min(min_x, df["nfkb_flat"].min())
        max_x = max(max_x, df["nfkb_flat"].max())

        valid_runs.append(df)

    start_x = df["nfkb_flat"].min()
    end_x = df["nfkb_flat"].max()

    print(f"Interpolating on range [{start_x}, {end_x}] with {n_points} points.")

    x_grid = np.linspace(start_x, end_x, n_points)
    interpolated_ys = []

    for df in valid_runs:
        y_interp = np.interp(x_grid, df["nfkb_flat"].values, df["pred_synth_factor"].values)
        interpolated_ys.append(y_interp)

    interpolated_ys = np.array(interpolated_ys)

    y_mean = np.mean(interpolated_ys, axis=0)
    y_std = np.std(interpolated_ys, axis=0)

    return x_grid, y_mean, y_std, interpolated_ys

def run_symbolic_regression_multi(experiment_dir):
    print(f"Processing experiment: {experiment_dir}")

    x_grid, y_mean, y_std, all_ys = load_and_aggregate_data(experiment_dir)

    X_df = pd.DataFrame(x_grid.reshape(-1, 1), columns=["x"])
    y = y_mean

    # Inverse variance weighting (epsilon guards against zero std)
    weights = 1.0 / (y_std**2 + 1e-10)

    print("Configuring Hill-type TemplateExpression for PySR...")

    # Hill-type template: p[1] + p[2] * f(x)^p[4] / (p[3] + f(x)^p[4])
    #   p[1] = offset
    #   p[2] = scale  (> 0: increasing Hill, < 0: decreasing Hill)
    #   p[3] = K = Kd^n  (folded constant; true Kd = p[3]^(1/p[4]))
    #   p[4] = n          (Hill exponent)
    # With binary_operators=["+"] only and high parsimony, f(x) should collapse to x.
    expression_spec = TemplateExpressionSpec(
        expressions=["f"],
        variable_names=["x"],
        parameters={"p": 4},
        combine="p[1] + p[2] * f(x)^p[4] / (p[3] + f(x)^p[4])",
    )

    model = PySRRegressor(
        niterations=10,
        binary_operators=["*"],
        unary_operators=[],
        expression_spec=expression_spec,
        maxsize=7,
        parsimony=10,
        model_selection="best",
        temp_equation_file=True,
        verbosity=1,
        progress=True,
        random_state=0,
        deterministic=True,
        parallelism='serial',
    )

    print("Fitting Hill-type symbolic regression model...")
    model.fit(X_df, y, weights=weights)

    # Pick the simplest equation with valid Hill parameters (K > 0, n > 0).
    # model_selection="best" can return negative-K solutions that break the Hill interpretation.
    best_equation = None
    for _, row in model.equations_.sort_values("complexity").iterrows():
        p = parse_hill_params(row["equation"])
        if p and p["K"] > 0 and p["n"] > 0:
            best_equation = row
            break
    if best_equation is None:
        print("Warning: no valid Hill equation found (K>0, n>0). Falling back to model best.")
        best_equation = model.get_best()

    params = parse_hill_params(best_equation["equation"])

    print("\n" + "="*50)
    print("Raw PySR equation:", best_equation["equation"])
    if params:
        print(f"\nHill form: {params['form']}")
        print(f"  offset = {params['offset']:.4f}")
        print(f"  scale  = {params['scale']:.4f}")
        print(f"  n      = {params['n']:.4f}")
        print(f"  Kd     = {params['Kd']:.4f}  (after folding c={params['c']:.4f} into K)")
    print("="*50 + "\n")

    # Save results
    output_log = os.path.join(experiment_dir, "symbolic_hill_regression_multi_results.txt")
    with open(output_log, "w") as f:
        f.write(f"Experiment: {experiment_dir}\n\n")
        f.write("Template: offset + scale * f(x)^n / (K + f(x)^n)\n")
        f.write("  [f(x) discovered by PySR; c folded into Kd]\n\n")
        f.write(f"Raw PySR equation: {best_equation['equation']}\n\n")
        if params:
            f.write(f"Hill form:  {params['form']}\n")
            f.write(f"  offset  = {params['offset']:.6f}\n")
            f.write(f"  scale   = {params['scale']:.6f}\n")
            f.write(f"  n       = {params['n']:.6f}\n")
            f.write(f"  Kd      = {params['Kd']:.6f}\n")
            f.write(f"  [raw: c = {params['c']:.6f}, K = {params['K']:.6f}]\n\n")
        f.write(f"Model R²: {model.score(X_df, y):.6f}\n")
        f.write("\nAll equations found:\n")
        try:
            f.write(model.equations_.to_string())
        except:
            pass

    print(f"Results saved to {output_log}")

    # Plotting
    print("Generating plot...")
    plt.figure(figsize=(6, 4))

    q05 = np.percentile(all_ys, 5, axis=0)
    q95 = np.percentile(all_ys, 95, axis=0)
    plt.fill_between(x_grid, q05, q95, color='C0', alpha=0.3, label="Learned Crosstalk (5-95% Range)")
    plt.plot(x_grid, y_mean, color='C0', linewidth=1, label="Learned Crosstalk (Ensemble Mean)")

    y_sym = model.predict(X_df)
    if params:
        fit_label = (f"{params['form'].capitalize()} Hill\n"
                     f"offset={params['offset']:.2f}, scale={params['scale']:.2f}, "
                     f"Kd={params['Kd']:.2f}, n={params['n']:.2f}")
    else:
        fit_label = f"Hill Fit ({best_equation['equation']})"
    plt.plot(x_grid, y_sym, c='C3', ls='--', linewidth=1, label=fit_label)

    plt.xlabel("NF-kB values")
    plt.ylabel("Crosstalk Factor")
    plt.title("Hill-type Symbolic Regression on Learned Crosstalk")
    plt.legend(fontsize=7)
    plt.grid(True, alpha=0.3)
    plt.ylim(bottom=0, top=5)
    plt.tight_layout()

    plot_file = os.path.join(experiment_dir, "symbolic_hill_regression_multi_plot.png")
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
