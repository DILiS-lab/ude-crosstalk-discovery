"""
run_one_phase: shared logic for both the train and test phases in kfold CV.
Encapsulates: data slicing, init, pre-equilibration, training, artifact saving, plotting.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import equinox as eqx
import matplotlib
matplotlib.use("Agg")
from pathlib import Path

from utils.models import (
    ude_models,
    transform_params_functions,
    new_boundaries_p,
    ode_models_p53_without_nfkb,
    final_solution_format,
)
from utils.plot_functions import plot_training_loss, plot_data
from utils.differential_equations_functions import ode_solve_in_parallel, ude_solve_in_parallel
from utils.get_parameters import initialize_for_ude
from utils.ude_utils import train_ude, unconstrained_to_ode_space, get_learned_factor_values


def run_one_phase(
    phase_dir: Path,
    config: dict,
    project_root: Path,
    cell_indices: np.ndarray,
    full_nfkb: jnp.ndarray,
    full_p53: jnp.ndarray,
    time_points: jnp.ndarray,
    ude_model,
    key: jax.Array,
    freeze_nn: bool = False,
    pretrained_nn=None,
):
    """
    Run one CV phase (train or test) for a single (fold, seed).

    Parameters
    ----------
    phase_dir      : Directory to write all artifacts for this phase.
    config         : Full experiment config dict.
    project_root   : Repo root (for resolving data paths).
    cell_indices   : 1-D array of original cell column indices for this phase.
    full_nfkb      : Full NF-kB signal matrix (T x N_total).
    full_p53       : Full p53 matrix (T x N_total).
    time_points    : JAX array of observation time points.
    ude_model      : UDE model function from utils.models.ude_models.
    key            : JAX PRNGKey.
    freeze_nn      : If True, freeze the NN and train only ODE params / ICs.
    pretrained_nn  : Pre-trained SynthNN to inject when freeze_nn=True.

    Returns
    -------
    trained_vars, p53_preds, neural_net_final
    """
    phase_dir = Path(phase_dir)
    phase_dir.mkdir(parents=True, exist_ok=True)

    model_name = config["model_name"]
    ode_solver_params = config["ode_solver_params"]
    rtol = ode_solver_params["rtol"]
    atol = ode_solver_params["atol"]
    max_steps = ode_solver_params["max_steps"]
    dt0 = ode_solver_params["dt0"]
    stiff = ode_solver_params["stiff"]

    pre_equil_params = config.get("pre_equilibration_solver_params", ode_solver_params)
    pre_rtol = pre_equil_params.get("rtol", rtol)
    pre_atol = pre_equil_params.get("atol", atol)
    pre_max_steps = pre_equil_params.get("max_steps", max_steps)
    pre_dt0 = pre_equil_params.get("dt0", dt0)
    pre_stiff = pre_equil_params.get("stiff", stiff)

    n_epochs = config["n_epochs"]
    learning_rate = config["learning_rate"]
    batch_size = config["batch_size"]
    offset_factor = config.get("offset_factor", None)
    scaling_factor = config.get("scaling_factor", None)
    t0 = config["time_start"]
    dt = config["time_step"]

    # Slice data to this phase's cells
    nfkb_phase = full_nfkb[:, cell_indices]
    p53_phase = full_p53[:, cell_indices]
    n_samples = len(cell_indices)

    # Record which original cell columns belong to this phase
    np.savetxt(phase_dir / "cell_indices.csv", cell_indices, fmt="%d", delimiter=",")

    # Initialize ODE params and ICs
    initialization_method = config["init_method"]
    init_args = config["init_args"]
    key, init_key = jax.random.split(key)
    model_params, initial_conditions, min_p, max_p, min_ic, max_ic = initialize_for_ude[
        initialization_method
    ](init_args, project_root, phase_dir, n_samples, init_key)

    # Pre-equilibration
    pre_equilibration_end = config.get("pre_equilibration_end", None)
    if pre_equilibration_end:
        pre_eq_points = jnp.arange(t0, pre_equilibration_end + dt, dt)
        params_before_eq, model_params_after_eq = transform_params_functions[model_name](
            model_params
        )
        solution = ode_solve_in_parallel(
            ode_models_p53_without_nfkb[model_name],
            pre_eq_points,
            pre_max_steps,
            pre_dt0,
            pre_rtol,
            pre_atol,
            stiff_solver=pre_stiff,
        )(initial_conditions, params_before_eq, batch_size=batch_size)
        initial_conditions = solution[-1, :, :]
        min_p, max_p = new_boundaries_p[model_name](min_p, max_p)
    else:
        model_params_after_eq = model_params

    np.savetxt(
        phase_dir / "initial_conditions_after_pre_equilibration.csv",
        initial_conditions,
        delimiter=",",
        fmt="%.6f",
    )

    # nn_factory: returns the pretrained NN unchanged when freezing
    nn_factory = (lambda k: pretrained_nn) if (freeze_nn and pretrained_nn is not None) else None

    key, train_key = jax.random.split(key)
    trained_vars, epoch_losses = train_ude(
        model_params_after_eq,
        min_p,
        max_p,
        n_epochs,
        n_samples,
        batch_size,
        learning_rate,
        phase_dir,
        initial_conditions,
        p53_phase,
        nfkb_phase,
        time_points,
        ude_model,
        max_steps,
        dt0,
        rtol,
        atol,
        stiff,
        model_name,
        min_ic=None,
        max_ic=None,
        offset_factor=offset_factor,
        scaling_factor=scaling_factor,
        key=train_key,
        nn_factory=nn_factory,
        freeze_nn=freeze_nn,
    )

    # Save NN (will equal train/neural_network.eqx for test phase when frozen)
    neural_net_final = trained_vars["nn"]
    eqx.tree_serialise_leaves(phase_dir / "neural_network.eqx", neural_net_final)

    # Decode learned params and ICs
    learned_model_params = unconstrained_to_ode_space(trained_vars["model_params"], min_p, max_p)
    learned_scaling_params = trained_vars["scaling_params"]

    # Replicate train_ude's internal IC bound logic for decoding
    min_ic_actual = jnp.min(initial_conditions, axis=0) * 0.5
    current_max = jnp.max(initial_conditions, axis=0)
    max_ic_actual = jnp.where(current_max == 0, 1.0, current_max * 2.0)
    max_ic_actual = jnp.where(max_ic_actual <= min_ic_actual, min_ic_actual + 1e-6, max_ic_actual)
    learned_initial_conditions = unconstrained_to_ode_space(
        trained_vars["initial_conditions"], min_ic_actual, max_ic_actual
    )

    # Save learned params/ICs/scaling
    np.savetxt(
        phase_dir / "learned_model_parameters.csv",
        learned_model_params,
        delimiter=",",
        fmt="%.6f",
    )
    np.savetxt(
        phase_dir / "learned_initial_conditions.csv",
        learned_initial_conditions,
        delimiter=",",
        fmt="%.6f",
    )
    pd.DataFrame(
        {
            "offset": np.array(learned_scaling_params["offset"]),
            "scale": np.array(learned_scaling_params["scale"]),
        }
    ).to_csv(phase_dir / "learned_scaling_parameters.csv", index=False)

    # Generate predictions with the final NN
    solution = ude_solve_in_parallel(
        ude_model, time_points, max_steps, dt0, rtol, atol, stiff
    )(
        learned_initial_conditions,
        (learned_model_params, nfkb_phase),
        neural_net_final,
        batch_size=batch_size,
    )
    p53_preds = final_solution_format[model_name](
        solution,
        learned_scaling_params["offset"],
        learned_scaling_params["scale"],
    )

    # Save prediction matrices
    np.savetxt(phase_dir / "p53_predictions.csv", np.array(p53_preds), delimiter=",", fmt="%.6f")
    np.savetxt(phase_dir / "p53_true.csv", np.array(p53_phase), delimiter=",", fmt="%.6f")
    np.savetxt(
        phase_dir / "pointwise_error.csv",
        np.array(p53_preds) - np.array(p53_phase),
        delimiter=",",
        fmt="%.6f",
    )

    # Crosstalk factor values (raw NN output — no regression)
    nfkb_flat, pred_synth_factor = get_learned_factor_values(
        neural_net_final, nfkb_phase, time_points
    )
    pd.DataFrame(
        {
            "nfkb_flat": np.array(nfkb_flat),
            "pred_synth_factor": np.array(pred_synth_factor),
        }
    ).to_csv(phase_dir / "learned_crosstalk_factor_values.csv", index=False)

    # --- Plots ---
    plot_training_loss(epoch_losses, phase_dir, "training_loss", savefig=True, show_title=False)

    key, plot_key = jax.random.split(key)
    plot_data(
        time_points,
        p53_phase,
        phase_dir,
        "true_vs_predicted_p53",
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

    plot_data(
        nfkb_flat,
        pred_synth_factor,
        phase_dir,
        "learned_crosstalk_shape",
        "NFkB value",
        "Factor value",
        ["Predicted function"],
        savefig=True,
        legend=True,
        show_markers=False,
        show_title=False,
    )

    rmse_tp = jnp.sqrt(jnp.mean((p53_phase - p53_preds) ** 2, axis=1))
    plot_data(
        time_points,
        rmse_tp.reshape((-1, 1)),
        phase_dir,
        "rmse_per_timepoint",
        "Time [h]",
        "RMSE",
        ["RMSE"],
        savefig=True,
        legend=False,
        show_title=False,
        ylim=(0, None),
    )

    return trained_vars, p53_preds, neural_net_final
