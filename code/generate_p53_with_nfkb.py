import jax
import sys
import json
from pathlib import Path
from datetime import datetime
import jax.numpy as jnp
import numpy as np
import pandas as pd

from utils.get_parameters import random_sample_parameters
from utils.models import (
    ode_models_p53_without_nfkb,
    ode_models_p53_with_nfkb,
    transform_params_functions,
    final_solution_format,
)
from utils.differential_equations_functions import (
    ode_solve_in_parallel,
    ode_solve_in_parallel_with_nfkb,
)
from utils.plot_functions import plot_data

jax.config.update("jax_enable_x64", True)

print("Setting up the experiment")
project_root = Path(__file__).resolve().parent

config_file_path = sys.argv[1]
with open(config_file_path, "r") as f:
    config = json.load(f)

if "output_folder" in config:
    experiment_folder = Path(config["output_folder"])
else:
    experiment_name = f"synthetic_data_{config['model_name']}_nfkb_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}"
    experiment_folder = Path(project_root) / "experiments" / experiment_name

experiment_folder.mkdir(parents=True, exist_ok=True)
print(f"Experiment folder created at: {experiment_folder}")

print("Extracting parameters and data from config file.")

t0 = config["time_start"]
t1 = config["time_end"]
dt = config["time_step"]
time_points = jnp.arange(t0, t1 + dt, dt)
model_name = config["model_name"]
func_type = config["crosstalk_function_type"]
ode_model_without_nfkb = ode_models_p53_without_nfkb[model_name]
ode_model_with_nfkb = ode_models_p53_with_nfkb[model_name]
rtol, atol, max_steps, dt0, stiff, batch_size = list(
    config["ode_solver_params"].values()
)
n_samples = config["n_samples"]
nfkb_signal = jnp.array(
    pd.read_csv(
        Path(project_root) / config["nfkb_signal_path"], header=None, index_col=False
    ).values
)

seed = config["seed"]
print(f"Random seed set to: {seed}")
np.random.seed(seed)
key = jax.random.PRNGKey(seed)

# Determine parameter loading mode
# If model_parameters is a list and n_samples > 1, we generate fresh parameters.
generate_fresh_params = False
if isinstance(config["model_parameters"], list) and n_samples > 1:
    generate_fresh_params = True

if generate_fresh_params:
    print(f"Generating {n_samples} fresh parameter sets based on seed {seed}.")
    base_params = jnp.array(config["model_parameters"])
    base_ic = jnp.array(config["initial_conditions"])
    change_scale = config.get("change_scale", 0.1)
    sample_initial_conditions = config.get("sample_initial_conditions", True)

    model_params, initial_conditions = random_sample_parameters(
        base_params,
        base_ic,
        n_samples,
        change_scale,
        sample_initial_conditions,
        experiment_folder,
        key,
    )
    p53_values_without_nfkb = None  # Will generate later

elif n_samples == 1:
    model_params = jnp.array(config["model_parameters"])
    initial_conditions = jnp.array(config["initial_conditions"])
    
    # Handle case where they might be paths (strings) even for n=1
    if isinstance(config["model_parameters"], str):
         model_params = jnp.array(pd.read_csv(Path(project_root) / config["model_parameters"], header=None, index_col=False).values)
    if isinstance(config["initial_conditions"], str):
         initial_conditions = jnp.array(pd.read_csv(Path(project_root) / config["initial_conditions"], header=None, index_col=False).values)

    # Ensure shape (1, D)
    if model_params.ndim == 1:
        model_params = jnp.broadcast_to(model_params, (1, model_params.shape[0]))
    elif model_params.shape[0] != 1:
        model_params = model_params[0:1] # Take first if multiple

    if initial_conditions.ndim == 1:
        initial_conditions = jnp.broadcast_to(
            initial_conditions, (1, initial_conditions.shape[0])
        )
    elif initial_conditions.shape[0] != 1:
        initial_conditions = initial_conditions[0:1]

    # Load comparison data if available
    if "p53_vals_without_nfkb_path" in config:
        p53_values_without_nfkb = jnp.array(
            pd.read_csv(
                Path(project_root) / config["p53_vals_without_nfkb_path"],
                header=None,
                index_col=False,
            ).values
        )
    else:
        p53_values_without_nfkb = None

else:
    # Load from files provided in config
    print("Loading parameters from provided files.")
    model_params = jnp.array(
        pd.read_csv(
            Path(project_root) / config["model_parameters"],
            header=None,
            index_col=False,
        ).values
    )
    initial_conditions = jnp.array(
        pd.read_csv(
            Path(project_root) / config["initial_conditions"],
            header=None,
            index_col=False,
        ).values
    )
    
    # Resample or slice parameters and initial conditions to match n_samples
    current_n = model_params.shape[0]
    if current_n != n_samples:
        print(f"Resampling parameters and initial conditions from {current_n} to {n_samples}")
        key, subkey = jax.random.split(key)

        # If we have more samples than needed, sample without replacement (subset)
        # If we have fewer samples than needed, sample with replacement (bootstrap)
        replace = n_samples > current_n

        idxs = jax.random.choice(subkey, current_n, shape=(n_samples,), replace=replace)

        model_params = model_params[idxs]
        initial_conditions = initial_conditions[idxs]

    # Load comparison data
    if "p53_vals_without_nfkb_path" in config:
        p53_values_without_nfkb = jnp.array(
            pd.read_csv(
                Path(project_root) / config["p53_vals_without_nfkb_path"],
                header=None,
                index_col=False,
            ).values
        )
        # We should also resample this if we resampled params!
        if current_n != n_samples and p53_values_without_nfkb.shape[1] == current_n:
             p53_values_without_nfkb = p53_values_without_nfkb[:, idxs]
    else:
        p53_values_without_nfkb = None


print("Solving the ODE to generate p53 signals with NFkB")

pre_equilibration = config.get("pre_equilibration_end", None)
if pre_equilibration:
    pre_equilibration_points = jnp.arange(t0, pre_equilibration + dt, dt)
    model_params_before_equilibration, model_params_after_equilibration = (
        transform_params_functions[model_name](model_params)
    )
    solution = ode_solve_in_parallel(
        ode_model_without_nfkb,
        pre_equilibration_points,
        max_steps,
        dt0,
        rtol,
        atol,
        stiff_solver=stiff,
    )(initial_conditions, model_params_before_equilibration, batch_size=batch_size)
    initial_conditions = solution[
        -1, :, :
    ]  # use final state of equilibration as initial conditions for the system after DNA damage

else:
    model_params_after_equilibration = model_params

params_with_signals = (model_params_after_equilibration, nfkb_signal)

solution = ode_solve_in_parallel_with_nfkb(
    ode_model_with_nfkb,
    time_points,
    max_steps,
    func_type,
    dt0,
    rtol,
    atol,
    stiff_solver=stiff,
)(initial_conditions, params_with_signals, batch_size=batch_size)

# Generate p53 without NFkB if needed (for fresh parameters)
if p53_values_without_nfkb is None:
    print("Generating p53 values without NFkB for comparison...")
    solution_no_nfkb = ode_solve_in_parallel(
        ode_model_without_nfkb,
        time_points,
        max_steps,
        dt0,
        rtol,
        atol,
        stiff_solver=stiff,
    )(initial_conditions, model_params_after_equilibration, batch_size=batch_size)

    if "offset_factor" in config and "scaling_factor" in config:
        p53_values_without_nfkb = final_solution_format[model_name](
            solution_no_nfkb, config["offset_factor"], config["scaling_factor"]
        )
    else:
        p53_values_without_nfkb = final_solution_format[model_name](solution_no_nfkb)

# return the values as expected by the specific model
if "offset_factor" in config and "scaling_factor" in config:
    # only for the Konrath 2020 ODE
    p53_values_with_nfkb = final_solution_format[model_name](
        solution, config["offset_factor"], config["scaling_factor"]
    )
else:
    p53_values_with_nfkb = final_solution_format[model_name](solution)

# Create a clean copy for plotting comparison later (optional, but good for clarity)
p53_values_clean = p53_values_with_nfkb

# Add measurement noise if specified
measurement_noise = config.get("measurement_noise", 0.0)
if measurement_noise > 0.0:
    print(f"Adding measurement noise with scale {measurement_noise}")
    # Regenerate key just in case
    key, noise_key = jax.random.split(key)
    # Use Log-Normal noise for strictly positive values emulating multiplicative sensor noise
    # p53_obs = p53_true * exp(N(0, sigma))
    eps = jax.random.normal(noise_key, shape=p53_values_with_nfkb.shape) * measurement_noise
    p53_values_with_nfkb = p53_values_with_nfkb * jnp.exp(eps)

# plot the generated data
plot_data(
    time_points,
    p53_values_with_nfkb,
    experiment_folder,
    "Synthetic generation of P53 dynamics",
    "Time [h]",
    "P53 level",
    ["Generations"],
    savefig=True,
    legend=False,
    plot_random_samples=True,
    show_title=False,
    key=key,
)
plot_data(
    time_points,
    p53_values_without_nfkb,
    experiment_folder,
    "Comparison of P53 dynamics with and without NFKB signal",
    "Time [h]",
    "P53 level",
    ["p53 without NFkB", "p53 with NFkB"],
    savefig=True,
    legend=True,
    data_2=p53_values_with_nfkb,
    plot_random_samples=True,
    show_title=False,
    key=key,
)


print("Saving generated P53 values to CSV file.")
np.savetxt(
    Path(experiment_folder) / "p53_generated.csv",
    p53_values_with_nfkb,
    delimiter=",",
    fmt="%.6f",
)

print("Synthetic data generation with NFkB signal finished.")
