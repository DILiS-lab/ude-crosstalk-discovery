import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import json
import sys
from pathlib import Path
import pandas as pd

from utils.neural_networks import SynthNN
from utils.models import (
    ude_models,
    final_solution_format,
)
from utils.differential_equations_functions import (
    ude_solve_in_parallel,
)


def load_ude_model(experiment_folder):
    """
    Loads a trained UDE model (neural network and ODE parameters) from an experiment folder.

    Args:
        experiment_folder: Path to the experiment folder containing the saved files.

    Returns:
        (neural_net, learned_params, config)
    """
    experiment_folder = Path(experiment_folder)

    # load configuration
    config_path = experiment_folder / "config.json"
    with open(config_path, "r") as f:
        config = json.load(f)
    model_name = config["model_name"]

    print(f"Loading model: {model_name}")

    # load the neural network
    key = jax.random.PRNGKey(0)
    model = SynthNN(key)

    # load the weights
    nn_path = experiment_folder / "neural_network.eqx"
    neural_net = eqx.tree_deserialise_leaves(nn_path, model)

    # load the learned ODE parameters
    params_path = experiment_folder / "learned_model_parameters.csv"
    learned_params = jnp.array(np.loadtxt(params_path, delimiter=","))

    return neural_net, learned_params, config


def solve_loaded_ude(
    neural_net, learned_params, config, t_steps, initial_conditions, nfkb_signal
):
    """
    Solves the UDE using the loaded model and parameters.

    Args:
        neural_net: The loaded Equinox neural network.
        learned_params: The loaded ODE parameters.
        config: The configuration dictionary.
        t_steps: Array of time points.
        initial_conditions: Initial conditions for the ODE.
        nfkb_signal: NF-kB signal array.

    Returns:
        The solution of the UDE.
    """

    model_name = config["model_name"]
    ode_solver_params = config["ode_solver_params"]
    ude_model = ude_models[model_name]
    max_steps = ode_solver_params["max_steps"]
    dt0 = ode_solver_params["dt0"]
    rtol = ode_solver_params["rtol"]
    atol = ode_solver_params["atol"]
    stiff = ode_solver_params["stiff"]
    offset_factor = config.get("offset_factor", None)
    scaling_factor = config.get("scaling_factor", None)

    solver = ude_solve_in_parallel(
        ude_model, t_steps, max_steps, dt0, rtol, atol, stiff
    )
    solution = solver(
        initial_conditions,
        (learned_params, nfkb_signal),
        neural_net,
        batch_size=10,  # Adjust batch size as needed
    )

    if offset_factor is not None and scaling_factor is not None:
        # only for the Konrath 2020 ODE
        p53_preds = final_solution_format[model_name](
            solution, offset_factor, scaling_factor
        )
    else:
        p53_preds = final_solution_format[model_name](solution)

    return p53_preds


if __name__ == "__main__":
    if len(sys.argv) > 1:
        experiment_path = sys.argv[1]

        try:
            nn, params, config = load_ude_model(experiment_path)
            print("Model loaded successfully.")
            print(f"Neural Net: {nn}")
            print(f"Params shape: {params.shape}")
            print(f"Config: {config.keys()}")

            # Reconstruct time points
            t0 = config["time_start"]
            t1 = config["time_end"]
            dt = config["time_step"]
            t_steps = jnp.arange(t0, t1 + dt, dt)

            # Load NF-kB signal
            project_root = Path(__file__).resolve().parent
            nfkb_signal = jnp.array(
                pd.read_csv(
                    Path(project_root) / config["nfkb_signal_path"],
                    header=None,
                    index_col=False,
                ).values
            )

            # Load initial conditions
            ic_path = (
                Path(experiment_path) / "initial_conditions_after_pre_equilibration.csv"
            )
            initial_conditions = jnp.array(np.loadtxt(ic_path, delimiter=","))
            print(f"Initial conditions loaded from {ic_path}")

            # Solve the UDE
            print("Solving UDE with loaded model")
            p53_preds = solve_loaded_ude(
                nn, params, config, t_steps, initial_conditions, nfkb_signal
            )

            # Save predictions
            output_path = Path(experiment_path) / "loaded_model_predictions.csv"
            np.savetxt(output_path, p53_preds, delimiter=",", fmt="%.6f")
            print(f"Predictions saved to {output_path}")

        except Exception as e:
            print(f"Error loading model: {e}")
    else:
        print("Please provide the experiment folder path as an argument.")
