import sys
import json
import jax
import numpy as np
import jax.numpy as jnp
from pathlib import Path
from datetime import datetime

from utils.plot_functions import plot_data
from utils.get_parameters import random_sample_parameters
from utils.models import nfkb_signal_models
from utils.differential_equations_functions import ode_solve_in_parallel

jax.config.update("jax_enable_x64", True)

print("Setting up the experiment")
project_root = Path(__file__).resolve().parent

config_file_path = sys.argv[1]
with open(config_file_path, "r") as f:
    config = json.load(f)

experiment_name = f"synthetic_data_{config['model_name']}_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}"
experiment_folder = Path(project_root) / "experiments" / experiment_name
experiment_folder.mkdir(parents=True, exist_ok=True)
print(f"Experiment folder created at: {experiment_folder}")

print("Extracting parameters and data from config file.")

t0 = config["time_start"]
t1 = config["time_end"]
dt = config["time_step"]
time_points = jnp.arange(t0, t1 + dt, dt)
initial_conditions = jnp.array(config["initial_conditions"])
ode_model = nfkb_signal_models[config["model_name"]]
model_params = jnp.array(config["model_parameters"])
dt0 = config["solver_dt0"]
max_steps = config["solver_max_steps"]

n_signals = config["n_signals"]

seed = config["seed"]
print(f"Random seed set to: {seed}")
np.random.seed(seed)
key = jax.random.PRNGKey(seed)

if n_signals == 1:
    model_params = jnp.broadcast_to(model_params, (1, model_params.shape[0]))
    initial_conditions = jnp.broadcast_to(
        initial_conditions, (1, initial_conditions.shape[0])
    )

else:
    print("Sampling parameters/initial conditions for multiple signals.")
    change_scale = config["change_scale"]
    sample_initial_conditions = config["sample_initial_conditions"]

    model_params, initial_conditions = random_sample_parameters(
        model_params,
        initial_conditions,
        n_signals,
        change_scale,
        sample_initial_conditions,
        experiment_folder,
        key,
    )

print("Solving the ODE to generate NFkB signal.")
solution = ode_solve_in_parallel(ode_model, time_points, max_steps, dt0)(
    initial_conditions, model_params
)

nfkb_generated = solution[:, :, 1]
# from the paper, the value of N is nuclear NFkB/total NFkB. We need the nuclear NFkB/cytosolic NFKB, so we convert: N/(1-N)
nfkb_generated = nfkb_generated / (1 - nfkb_generated)

plot_data(
    time_points,
    nfkb_generated,
    experiment_folder,
    "Synthetic generation of NFkB signal",
    "Time [h]",
    "NFkB level",
    ["Generations"],
    savefig=True,
    legend=False,
    show_title=False,
)

print("Saving generated NFkB signal.")
np.savetxt(
    Path(experiment_folder) / "nfkb_generated.csv",
    nfkb_generated,
    delimiter=",",
    fmt="%.6f",
)

print("NFkB signal generation finished.")
