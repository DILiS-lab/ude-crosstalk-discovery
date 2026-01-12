import jax
import sys
import json
from pathlib import Path
from datetime import datetime
import jax.numpy as jnp
import numpy as np

from utils.get_parameters import random_sample_parameters
from utils.models import (
    ode_models_p53_without_nfkb,
    transform_params_functions,
    final_solution_format,
)
from utils.differential_equations_functions import ode_solve_in_parallel
from utils.plot_functions import plot_data

jax.config.update("jax_enable_x64", True)

print("Setting up the experiment")
project_root = Path(__file__).resolve().parent

config_file_path = sys.argv[1]
with open(config_file_path, "r") as f:
    config = json.load(f)

experiment_name = f"synthetic_data_{config['model_name']}_no_nfkb_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}"
experiment_folder = Path(project_root) / "experiments" / experiment_name
experiment_folder.mkdir(parents=True, exist_ok=True)
print(f"Experiment folder created at: {experiment_folder}")

print("Extracting parameters and data from config file.")

t0 = config["time_start"]
t1 = config["time_end"]
dt = config["time_step"]
time_points = jnp.arange(t0, t1 + dt, dt)
initial_conditions = jnp.array(config["initial_conditions"])
model_name = config["model_name"]
ode_model = ode_models_p53_without_nfkb[model_name]
model_params = jnp.array(config["model_parameters"])
rtol, atol, max_steps, dt0, stiff, batch_size = list(
    config["ode_solver_params"].values()
)
n_samples = config["n_samples"]

if n_samples == 1:
    model_params = jnp.broadcast_to(model_params, (1, model_params.shape[0]))
    initial_conditions = jnp.broadcast_to(
        initial_conditions, (1, initial_conditions.shape[0])
    )

    seed = config["seed"]
    print(f"Random seed set to: {seed}")
    np.random.seed(seed)
    key = jax.random.PRNGKey(seed)

else:
    print("Sampling parameters/ initial states for multiple samples.")
    change_scale = config["change_scale"]
    sample_initial_conditions = config["sample_initial_conditions"]

    seed = config["seed"]
    print(f"Random seed set to: {seed}")
    np.random.seed(seed)
    key = jax.random.PRNGKey(seed)

    model_params, initial_conditions = random_sample_parameters(
        model_params,
        initial_conditions,
        n_samples,
        change_scale,
        sample_initial_conditions,
        experiment_folder,
        key,
    )

print("Solving the ODE to generate p53 signals without NFkB")

pre_equilibration = config.get("pre_equilibration_end", None)
if pre_equilibration:
    pre_equilibration_points = jnp.arange(t0, pre_equilibration + dt, dt)
    model_params_before_equilibration, model_params_after_equilibration = (
        transform_params_functions[model_name](model_params)
    )
    solution = ode_solve_in_parallel(
        ode_model,
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

solution = ode_solve_in_parallel(
    ode_model, time_points, max_steps, dt0, rtol, atol, stiff_solver=stiff
)(initial_conditions, model_params_after_equilibration, batch_size=batch_size)

# return the values as expected by the specific model
if "offset_factor" in config and "scaling_factor" in config:
    # only for the Konrath 2020 ODE
    p53_values = final_solution_format[model_name](
        solution, config["offset_factor"], config["scaling_factor"]
    )
else:
    p53_values = final_solution_format[model_name](solution)

# plot the generated generated data
plot_data(
    time_points,
    p53_values,
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

print("Saving generated P53 values to CSV file.")
np.savetxt(
    Path(experiment_folder) / "p53_generated.csv", p53_values, delimiter=",", fmt="%.6f"
)

print("Synthetic data generation without NFkB signal finished.")
