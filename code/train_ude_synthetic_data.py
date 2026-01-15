import jax
import json
import sys
import pickle
import jax.numpy as jnp
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import equinox as eqx

from utils.models import (
    ude_models,
    transform_params_functions,
    new_boundaries_p,
    ode_models_p53_without_nfkb,
    final_solution_format,
)
from utils.plot_functions import (
    plot_training_loss,
    plot_data,
    compute_and_plot_rmse_per_timepoint,
)
from utils.differential_equations_functions import (
    ode_solve_in_parallel,
    ude_solve_in_parallel,
)
from utils.get_parameters import initialize_for_ude
from utils.ude_utils import (
    train_ude,
    unconstrained_to_ode_space,
    calculate_and_print_rmse_per_parameter,
    get_true_vs_learned_factor_values,
)
from utils.assumed_crosstalk_functions import get_assumed_crosstalk_function
from utils.symbolic_regression import run_symbolic_regression

jax.config.update("jax_enable_x64", True)

print("Setting up the experiment")
project_root = Path(__file__).resolve().parent

config_file_path = sys.argv[1]
with open(config_file_path, "r") as f:
    config = json.load(f)

experiment_name = (
    f"ude_{config['model_name']}_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}"
)
experiment_folder = Path(project_root) / "experiments" / experiment_name
experiment_folder.mkdir(parents=True, exist_ok=True)
print(f"Experiment folder created at: {experiment_folder}")

# Save the configuration used for this experiment
with open(experiment_folder / "config.json", "w") as f:
    json.dump(config, f, indent=4)

print("Extracting parameters and data from config file.")
t0 = config["time_start"]
t1 = config["time_end"]
dt = config["time_step"]
time_points = jnp.arange(t0, t1 + dt, dt)
model_name = config["model_name"]
ude_model = ude_models[model_name]
ode_solver_params = config["ode_solver_params"]
rtol = ode_solver_params["rtol"]
atol = ode_solver_params["atol"]
max_steps = ode_solver_params["max_steps"]
dt0 = ode_solver_params["dt0"]
stiff = ode_solver_params["stiff"]

pre_equil_solver_params = config.get(
    "pre_equilibration_solver_params", ode_solver_params
)
pre_rtol = pre_equil_solver_params.get("rtol", rtol)
pre_atol = pre_equil_solver_params.get("atol", atol)
pre_max_steps = pre_equil_solver_params.get("max_steps", max_steps)
pre_dt0 = pre_equil_solver_params.get("dt0", dt0)
pre_stiff = pre_equil_solver_params.get("stiff", stiff)
nfkb_signal = jnp.array(
    pd.read_csv(
        Path(project_root) / config["nfkb_signal_path"], header=None, index_col=False
    ).values
)
true_p53_values = jnp.array(
    pd.read_csv(
        Path(project_root) / config["true_p53_values_path"],
        header=None,
        index_col=False,
    ).values
)
n_epochs = config["n_epochs"]
learning_rate = config["learning_rate"]
batch_size = config["batch_size"]
n_samples = true_p53_values.shape[1]
offset_factor = config.get("offset_factor", None)
scaling_factor = config.get("scaling_factor", None)
func_type = config["crosstalk_function_type"]

seed = config["seed"]
print(f"Random seed set to: {seed}")
np.random.seed(seed)
key = jax.random.PRNGKey(seed)
init_key, train_key, plot_key, sr_key = jax.random.split(key, 4)

initialization_method, init_args = config["init_method"], config["init_args"]
model_params, initial_conditions, min_p, max_p, min_ic, max_ic = initialize_for_ude[
    initialization_method
](init_args, project_root, experiment_folder, n_samples, init_key)

pre_equilibration = config.get("pre_equilibration_end", None)
if pre_equilibration:
    print("Running pre-equilibration phase")
    pre_equilibration_points = jnp.arange(t0, pre_equilibration + dt, dt)
    model_params_before_equilibration, model_params_after_equilibration = (
        transform_params_functions[model_name](model_params)
    )
    solution = ode_solve_in_parallel(
        ode_models_p53_without_nfkb[model_name],
        pre_equilibration_points,
        pre_max_steps,
        pre_dt0,
        pre_rtol,
        pre_atol,
        stiff_solver=pre_stiff,
    )(initial_conditions, model_params_before_equilibration, batch_size=batch_size)
    initial_conditions = solution[
        -1, :, :
    ]  # use final state of equilibration as initial conditions for the system after DNA damage
    min_p, max_p = new_boundaries_p[model_name](min_p, max_p)
else:
    model_params_after_equilibration = model_params

np.savetxt(
    Path(experiment_folder) / "initial_conditions_after_pre_equilibration.csv",
    initial_conditions,
    delimiter=",",
    fmt="%.6f",
)

print(
    f"Initial conditions saved to {experiment_folder}/initial_conditions_after_pre_equilibration.csv"
)

# Define initial condition bounds for training (simple buffer around current ICs)
min_ic_train = None
max_ic_train = None

print("Starting the training of the UDE model")
trained_vars, epoch_losses = train_ude(
    model_params_after_equilibration,
    min_p,
    max_p,
    n_epochs,
    n_samples,
    batch_size,
    learning_rate,
    experiment_folder,
    initial_conditions,
    true_p53_values,
    nfkb_signal,
    time_points,
    ude_model,
    max_steps,
    dt0,
    rtol,
    atol,
    stiff,
    model_name,
    min_ic=min_ic_train,
    max_ic=max_ic_train,
    offset_factor=offset_factor,
    scaling_factor=scaling_factor,
    key=train_key,
)
plot_training_loss(
    epoch_losses,
    experiment_folder,
    "Training Loss over Epochs",
    savefig=True,
    show_title=False,
)

print("Training completed.")

# get final learned neural network and save its parameters
neural_net_final = trained_vars["nn"]
eqx.tree_serialise_leaves(
    Path(experiment_folder) / "neural_network.eqx", neural_net_final
)

learned_model_params = unconstrained_to_ode_space(
    trained_vars["model_params"], min_p, max_p
)

# Extract learned initial conditions
# We need to reconstruct min_ic and max_ic if we didn't pass them, to decode.
# Since we passed None, train_ude used internal defaults.
min_ic_train = jnp.min(initial_conditions, axis=0) * 0.5
current_max = jnp.max(initial_conditions, axis=0)
max_ic_train = jnp.where(current_max == 0, 1.0, current_max * 2.0)
max_ic_train = jnp.where(max_ic_train <= min_ic_train, min_ic_train + 1e-6, max_ic_train)

learned_initial_conditions = unconstrained_to_ode_space(
    trained_vars["initial_conditions"], min_ic_train, max_ic_train
)


# Calculate and print RMSE between learned and initial parameters
print("Parameter Learning Results:")
calculate_and_print_rmse_per_parameter(
    learned_model_params, model_params_after_equilibration
)

np.savetxt(
    Path(experiment_folder) / "learned_model_parameters.csv",
    learned_model_params,
    delimiter=",",
    fmt="%.6f",
)
print(f"Learned parameters saved to {experiment_folder}/learned_parameters.csv")

np.savetxt(
    Path(experiment_folder) / "learned_initial_conditions.csv",
    learned_initial_conditions,
    delimiter=",",
    fmt="%.6f",
)
print(f"Learned initial conditions saved to {experiment_folder}/learned_initial_conditions.csv")

print("Generating p53 predictions with trained UDE model.")
solution = ude_solve_in_parallel(
    ude_model, time_points, max_steps, dt0, rtol, atol, stiff
)(
    learned_initial_conditions,
    (learned_model_params, nfkb_signal),
    neural_net_final,
    batch_size=batch_size,
)

if offset_factor is not None and scaling_factor is not None:
    # only for the Konrath 2020 ODE
    p53_preds = final_solution_format[model_name](
        solution, offset_factor, scaling_factor
    )
else:
    p53_preds = final_solution_format[model_name](solution)

plot_data(
    time_points,
    true_p53_values,
    experiment_folder,
    "True vs Predicted Values of p53 levels",
    "Time [h]",
    "P53 level",
    ["True values", "Predictions"],
    savefig=True,
    legend=True,
    data_2=p53_preds,
    plot_random_samples=True,
    show_title=False,
    key=plot_key,
)

# Compute and plot RMSE per time point between true and predicted p53 values
compute_and_plot_rmse_per_timepoint(
    true_p53_values, p53_preds, time_points, experiment_folder
)

print("Generating true vs learned crosstalk function values.")
nfkb_flat, true_synth_factor, pred_synth_factor = get_true_vs_learned_factor_values(
    neural_net_final,
    get_assumed_crosstalk_function,
    nfkb_signal,
    time_points,
    func_type,
)

# Save learned factor values to CSV
learned_factor_df = pd.DataFrame(
    {
        "nfkb_flat": np.array(nfkb_flat),
        "true_synth_factor": np.array(true_synth_factor),
        "pred_synth_factor": np.array(pred_synth_factor),
    }
)
learned_factor_df.to_csv(
    Path(experiment_folder) / "learned_vs_true_crosstalk_factor_values.csv", index=False
)
print(
    f"Learned factor values saved to {experiment_folder}/learned_vs_true_crosstalk_factor_values.csv"
)

plot_data(
    nfkb_flat,
    true_synth_factor,
    experiment_folder,
    "True vs Learned P53 - NF-κB Crosstalk Shape",
    "NFkB value",
    "Factor value",
    ["Assumed function", "Predicted function"],
    savefig=True,
    legend=True,
    data_2=pred_synth_factor,
    show_markers=True,
    show_title=False,
)

print("Running symbolic regression to extract learned crosstalk function.")
symbolic_reg_function, s_r_params, symbolic_model = run_symbolic_regression(
    experiment_folder=experiment_folder,
    input=nfkb_flat.reshape((-1, 1)),
    predictions=pred_synth_factor.reshape((-1, 1)),
    key=sr_key,
)

# Save the symbolic regression model
with open(Path(experiment_folder) / "symbolic_regression_model.pkl", "wb") as f:
    pickle.dump(symbolic_model, f)
print(
    f"Symbolic regression model saved to {experiment_folder}/symbolic_regression_model.pkl"
)


def symbolic_reg_model(x):
    """
    Apply a symbolic regression model to input data. The function takes an input value or array, reshapes it to a 2D vector,
    and applies a symbolic regression function with pre-computed parameters.
    Args:
        x: Input value or array to be processed by the symbolic regression model.
    Returns:
        The squeezed output of the symbolic regression
    """

    x_arr = jnp.asarray(x)
    x_arr = jnp.reshape(x_arr, (-1, 1))
    result = symbolic_reg_function(x_arr, s_r_params)
    return jnp.squeeze(result)


print("Generating predictions with symbolic regression UDE model.")
solution = ude_solve_in_parallel(
    ude_model, time_points, max_steps, dt0, rtol, atol, stiff
)(
    initial_conditions,
    (learned_model_params, nfkb_signal),
    symbolic_reg_model,
    batch_size=batch_size,
)
if offset_factor is not None and scaling_factor is not None:
    # only for the Konrath 2020 ODE
    p53_symbolic = final_solution_format[model_name](
        solution, offset_factor, scaling_factor
    )
else:
    p53_symbolic = final_solution_format[model_name](solution)

plot_data(
    time_points,
    true_p53_values,
    experiment_folder,
    "True vs Predicted vs Symbolic Values of p53 levels",
    "Time [h]",
    "P53 level",
    ["True values", "Predictions", "Symbolic values"],
    savefig=True,
    legend=True,
    data_2=p53_preds,
    data_3=p53_symbolic,
    plot_random_samples=True,
    show_title=False,
)

crosstalk_symbolic = symbolic_reg_model(nfkb_flat.reshape((-1, 1)))
plot_data(
    nfkb_flat,
    true_synth_factor,
    experiment_folder,
    "True vs Learned vs Symbolic P53 - NF-kB Crosstalk Shape",
    "NFkB value",
    "Factor value",
    ["Assumed function", "Predicted function", "Symbolic function"],
    savefig=True,
    legend=True,
    data_2=pred_synth_factor,
    data_3=crosstalk_symbolic,
    show_markers=True,
    show_title=False,
)


def rmse_func(a, b):
    return jnp.sqrt(jnp.mean((a - b) ** 2))


rmse_true_pred = f"RMSE between true and predicted crosstalk function values: {rmse_func(true_synth_factor, pred_synth_factor):.6f}"
rmse_true_symb = f"RMSE between true and symbolic-regression crosstalk function values: {rmse_func(true_synth_factor, crosstalk_symbolic):.6f}"
rmse_pred_symb = f"RMSE between predicted and symbolic-regression crosstalk function values: {rmse_func(pred_synth_factor, crosstalk_symbolic):.6f}"
print(rmse_true_pred)
print(rmse_true_symb)
print(rmse_pred_symb)

with open(Path(experiment_folder) / "rmse_crosstalk_values.txt", "a") as f:
    f.write(rmse_true_pred + "\n")
    f.write(rmse_true_symb + "\n")
    f.write(rmse_pred_symb + "\n")

np.savetxt(
    Path(experiment_folder) / "p53_generated.csv", p53_preds, delimiter=",", fmt="%.6f"
)
