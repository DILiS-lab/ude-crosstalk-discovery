import math
from pathlib import Path
import equinox as eqx
import jax
import jax.numpy as jnp
from utils.differential_equations_functions import ude_solve_in_parallel
from utils.models import final_solution_format
from utils.neural_networks import SynthNN
import optax


def get_learned_factor_values(nn_model, nfkb_signal, t_obs):
    """
    Get learned factor values from the neural network model given NF-κB signal and time observations.
    Args:
        nn_model: The trained neural network model.
        nfkb_signal: NF-κB signal values.
        t_obs: Observed time points.
    Returns:
        A tuple containing sorted NF-κB signal values and their corresponding learned factor values.
    """

    n_gens = nfkb_signal.shape[1]
    batched_predictions = jax.vmap(lambda x: nn_model(x))

    return_matrix = jnp.zeros((n_gens * len(t_obs), 2))
    return_matrix = return_matrix.at[:, 0].set(jnp.array(nfkb_signal).T.ravel())
    return_matrix = return_matrix.at[:, 1].set(
        batched_predictions(return_matrix[:, 0].reshape(-1, 1))
    )

    sorted_combined = return_matrix[jnp.argsort(return_matrix[:, 0])]
    return (sorted_combined[:, 0], sorted_combined[:, 1])


def get_true_vs_learned_factor_values(
    nn_model, assumed_func, nfkb_signal, t_obs, func_type
):
    """
    Get true vs learned factor values from the neural network model and assumed function given NF-κB signal and time observations.
    Args:
        nn_model: The trained neural network model.
        assumed_func: The assumed crosstalk function.
        nfkb_signal: NF-κB signal values.
        t_obs: Observed time points.
        func_type: Type of crosstalk function to use ("sigmoid", "decreasing", "oscillatory").
    Returns:
        A tuple containing sorted NF-κB signal values, true factor values, and learned factor values.
    """

    n_gens = nfkb_signal.shape[1]
    batched_predictions = jax.vmap(lambda x: nn_model(x))

    return_matrix = jnp.zeros((n_gens * len(t_obs), 3))
    return_matrix = return_matrix.at[:, 0].set(jnp.array(nfkb_signal).T.ravel())
    return_matrix = return_matrix.at[:, 2].set(
        batched_predictions(return_matrix[:, 0].reshape(-1, 1))
    )

    for i in range(n_gens):
        func = assumed_func(t_obs, nfkb_signal[:, i], func_type)
        return_matrix = return_matrix.at[i * len(t_obs) : (i + 1) * len(t_obs), 1].set(
            jnp.array([func(t) for t in t_obs]).ravel()
        )

    sorted_combined = return_matrix[jnp.argsort(return_matrix[:, 0])]
    return (sorted_combined[:, 0], sorted_combined[:, 1], sorted_combined[:, 2])


def calculate_and_print_rmse_per_parameter(
    learned_params, true_params, param_names=None, experiment_folder=None
):
    """
    Calculate and print RMSE per parameter between learned and true parameters.
    Args:
        learned_params: Array of learned parameter values.
        true_params: Array of true parameter values.
        param_names: Optional list of parameter names for better readability.
        experiment_folder: Optional path to save the RMSE results.
    Returns:
        A tuple containing overall RMSE and RMSE per parameter.
    """

    rmse_overall = jnp.sqrt(jnp.mean((learned_params - true_params) ** 2))
    overall_msg = f"Overall RMSE: {rmse_overall:.6f}"
    print(overall_msg)

    rmse_per_column = jnp.sqrt(jnp.mean((learned_params - true_params) ** 2, axis=0))

    param_msgs = ["\nRMSE per parameter:"]
    for i, rmse_val in enumerate(rmse_per_column):
        if param_names and i < len(param_names):
            msg = f"  {param_names[i]}: {rmse_val:.6f}"
        else:
            msg = f"  Parameter {i}: {rmse_val:.6f}"
        param_msgs.append(msg)
        print(msg)

    # Save to file if experiment_folder is provided
    if experiment_folder is not None:
        rmse_file = Path(experiment_folder) / "rmse_params_results.txt"
        with open(rmse_file, "a") as f:
            f.write(overall_msg + "\n")
            for msg in param_msgs:
                f.write(msg + "\n")
            f.write("\n")

    return rmse_overall, rmse_per_column


def unconstrained_to_ode_space(params, p_min, p_max):
    """
    Transform unconstrained parameters to ODE parameter space using sigmoid scaling.
    Args:
        params: Array of unconstrained parameter values.
        p_min: Minimum bounds for parameters.
        p_max: Maximum bounds for parameters.
    Returns:
        Array of parameters transformed to ODE parameter space.
    """

    sig = jax.nn.sigmoid(params)
    return p_min + (p_max - p_min) * sig


def ode_space_to_unconstrained(params, p_min, p_max):
    """
    Transform ODE parameter space parameters to unconstrained space using inverse sigmoid scaling.
    Args:
        params: Array of ODE parameter values.
        p_min: Minimum bounds for parameters.
        p_max: Maximum bounds for parameters.
    Returns:
        Array of parameters transformed to unconstrained space.
    """

    eps = 1e-7
    scaled = (params - p_min) / (p_max - p_min)
    scaled = jnp.clip(scaled, eps, 1 - eps)
    return jnp.log(scaled / (1 - scaled))


@eqx.filter_jit
@eqx.filter_value_and_grad
def loss_fn(
    trainable_vars,
    ude_model,
    solver_fn,
    params_min,
    params_max,
    idxs,
    p53_true_batch,
    initial_conditions_batch,
    nfkb_signal_batch,
    model_name,
):
    """
    Compute the loss for a batch of data during UDE training.
    Args:
        trainable_vars: Dictionary containing trainable variables (neural network and model parameters).
        ude_model: The baseline ODE system.
        solver_fn: The UDE solver function.
        params_min: Minimum bounds for model parameters.
        params_max: Maximum bounds for model parameters.
        idxs: Indices of the current batch.
        p53_true_batch: True p53 values for the current batch.
        initial_conditions_batch: Initial conditions for the current batch.
        nfkb_signal_batch: NF-κB signals for the current batch.
        model_name: Name of the model being used.
    Returns:
        Computed loss value for the current batch.
    """

    synth_nn = trainable_vars["nn"]
    raw_model_params = trainable_vars["model_params"]
    raw_p_batch = raw_model_params[idxs]

    scaling_params = trainable_vars["scaling_params"]
    offset_batch = scaling_params["offset"][idxs]
    scaling_batch = scaling_params["scale"][idxs]

    contr_params_batch = unconstrained_to_ode_space(raw_p_batch, params_min, params_max)

    solution = solver_fn(
        initial_conditions_batch,
        (contr_params_batch, nfkb_signal_batch),
        synth_nn,
        None,
    )

    p53_preds_batch = final_solution_format[model_name](
            solution, offset_batch, scaling_batch
        )

    loss = jnp.mean(optax.huber_loss(p53_preds_batch, p53_true_batch, delta=1.0))

    return loss


@eqx.filter_jit
def training_step(
    trainable_vars,
    ude_model,
    solver_fn,
    opt_state,
    optimizer,
    idx_batch,
    initial_conditions_batch,
    p53_true_batch,
    nfkb_signal_batch,
    params_min,
    params_max,
    model_name,
    offset_factor=None,
    scaling_factor=None,
):
    """
    Perform a single training step for UDE model.
    Args:
        trainable_vars: Dictionary containing trainable variables (neural network and model parameters).
        ude_model: The baseline ODE system.
        solver_fn: The UDE solver function.
        opt_state: Current optimizer state.
        optimizer: Optax optimizer instance.
        idx_batch: Indices of the current batch.
        initial_conditions_batch: Initial conditions for the current batch.
        p53_true_batch: True p53 values for the current batch.
        nfkb_signal_batch: NF-κB signals for the current batch.
        params_min: Minimum bounds for model parameters.
        params_max: Maximum bounds for model parameters.
        model_name: Name of the model being used.
        offset_factor: Optional offset factor for formatting the solution.
        scaling_factor: Optional scaling factor for formatting the solution.
    Returns:
        Updated trainable variables, updated optimizer state, and computed loss for the current batch.
    """

    loss, grads = loss_fn(
        trainable_vars,
        ude_model,
        solver_fn,
        params_min,
        params_max,
        idx_batch,
        p53_true_batch,
        initial_conditions_batch,
        nfkb_signal_batch,
        model_name,
    )

    grads = jax.tree_util.tree_map(
        lambda g: jnp.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0), grads
    )
    
    updates, opt_state = optimizer.update(grads, opt_state, trainable_vars)
    trainable_vars = eqx.apply_updates(trainable_vars, updates)

    return trainable_vars, opt_state, loss


def train_ude(
    model_params,
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
    offset_factor=None,
    scaling_factor=None,
    key=None,
    nn_factory=None,
):
    """
    Train a UDE model using the provided data and parameters.
    Args:
        model_params: Initial model parameters.
        min_p: Minimum bounds for model parameters.
        max_p: Maximum bounds for model parameters.
        n_epochs: Number of training epochs.
        n_samples: Number of samples in the dataset.
        batch_size: Size of each training batch.
        learning_rate: Learning rate for the optimizer.
        experiment_folder: Folder path to save training logs.
        initial_conditions: Initial conditions for the ODE solver.
        true_p53_values: True p53 values for training.
        nfkb_signal: NF-κB signals for training.
        time_points: Time points for the ODE solver.
        ude_model: The baseline ODE system.
        max_steps: Maximum number of integration steps allowed.
        dt0: Initial time step size.
        rtol: Relative tolerance for the solver.
        atol: Absolute tolerance for the solver.
        stiff: Boolean indicating if a stiff solver should be used.
        model_name: Name of the model being used.
        offset_factor: Optional offset factor for formatting the solution.
        scaling_factor: Optional scaling factor for formatting the solution.
        key: JAX PRNGKey for random number generation.
        nn_factory: Optional factory callable to create the neural network. Defaults to SynthNN.
    Returns:
        A tuple containing:
            - trained_vars: Dictionary of trained variables (neural network and model parameters).
            - epoch_losses: List of average epoch losses over training.
    """

    if key is None:
        key = jax.random.PRNGKey(0)

    key, nn_key = jax.random.split(key)
    
    if nn_factory:
        synth_nn = nn_factory(nn_key)
    else:
        synth_nn = SynthNN(nn_key)

    params_unconstrained = ode_space_to_unconstrained(model_params, min_p, max_p)

    scaling_params = {
            "offset": jnp.full((n_samples,), offset_factor),
            "scale": jnp.full((n_samples,), scaling_factor),
        }

    trainable_vars = {"nn": synth_nn, 
                      "model_params": params_unconstrained,
                      "scaling_params": scaling_params}

    solver_fn = ude_solve_in_parallel(
        ude_model, time_points, max_steps, dt0, rtol, atol, stiff_solver=stiff
    )

    optim = optax.chain(
        optax.clip_by_global_norm(1.0),  # gradient clipping
        optax.adamw(learning_rate=learning_rate, weight_decay=0.01 * learning_rate),
    )

    opt_state = optim.init(eqx.filter(trainable_vars, eqx.is_array))

    N = n_samples
    epoch_losses = []

    log_file = Path(experiment_folder) / "training_log.txt"
    log_handle = open(log_file, "a", buffering=1)

    for epoch in range(n_epochs):
        key, perm_key = jax.random.split(key)
        perm = jax.random.permutation(perm_key, N)
        epoch_loss = 0.0
        num_batches = 0

        for i in range(0, N, batch_size):
            batch_end = min(i + batch_size, N)
            idx_batch = perm[i:batch_end]

            initial_conditions_batch = initial_conditions[idx_batch]
            true_p53_batch = true_p53_values[:, idx_batch]
            nfkb_signal_batch = nfkb_signal[:, idx_batch]

            trainable_vars, opt_state, l = training_step(
                trainable_vars,
                ude_model,
                solver_fn,
                opt_state,
                optim,
                idx_batch,
                initial_conditions_batch,
                true_p53_batch,
                nfkb_signal_batch,
                min_p,
                max_p,
                model_name,
                offset_factor,
                scaling_factor,
            )

            epoch_loss += l
            num_batches += 1

        avg_epoch_loss = epoch_loss / num_batches
        epoch_losses.append(avg_epoch_loss)

        log_handle.write(
            f"Epoch {epoch + 1}/{n_epochs}, Average Epoch Loss = {avg_epoch_loss:.6f}\n"
        )
        log_handle.flush()

        print(
            f"Epoch {epoch + 1}/{n_epochs}, Average Epoch Loss = {avg_epoch_loss:.6f}"
        )

    log_handle.close()

    return trainable_vars, epoch_losses
