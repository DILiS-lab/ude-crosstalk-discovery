from pathlib import Path
import pandas as pd
import os
import jax
import jax.numpy as jnp
import numpy as np


def random_sample_parameters(
    model_params,
    initial_conditions,
    n_signals,
    change_scale,
    sample_initial_conditions,
    experiment_folder,
    key,
):
    """
    Randomly sample model parameters and initial conditions using log-normal distribution.
    Args:
        model_params: Array of model parameters to sample around.
        initial_conditions: Array of initial conditions to sample around.
        n_signals: Number of trajectories/signals on which to sample.
        change_scale: Scale of variation for sampling.
        sample_initial_conditions: Boolean indicating whether to sample initial conditions.
        experiment_folder: Folder path to save sampled parameters and initial conditions.
        key: JAX random key for sampling.
    Returns:
        sampled_params: Array of shape (n_signals, n_params) with sampled model parameters.
        initial_conditions_: Array of shape (n_signals, n_initial_conditions) with sampled initial conditions.
    """
    n_params = model_params.shape[0]
    sampled_params = jnp.zeros((n_signals, n_params))
    sampled_params = sampled_params.at[0, :].set(model_params)

    for i in range(len(model_params)):
        key, subkey = jax.random.split(key)
        param = model_params[i]

        # calculate mu and sigma for log-normal distribution
        mu_x_sq = param**2
        std_x = param * change_scale
        var_x = std_x**2
        mu = np.log((mu_x_sq) / np.sqrt(var_x + mu_x_sq))
        sigma = np.sqrt(np.log(var_x / (mu_x_sq) + 1))
        
        # approximate lognormal using jax.random.normal
        # X ~ Lognormal(mu, sigma) <=> log(X) ~ Normal(mu, sigma)
        # <=> X = exp(mu + sigma * Z), where Z ~ Normal(0, 1)
        z = jax.random.normal(subkey, shape=(n_signals - 1,))
        sampled_values = jnp.exp(mu + sigma * z)
        
        sampled_params = sampled_params.at[1:, i].set(sampled_values)

    if sample_initial_conditions:
        n_init = initial_conditions.shape[0]
        initial_conditions_ = jnp.zeros((n_signals, n_init))
        initial_conditions_ = initial_conditions_.at[0, :].set(initial_conditions)

        for i in range(n_init):
            cond = initial_conditions[i]

            if cond > 0:
                key, subkey = jax.random.split(key)
                
                # calculate mu and sigma for log-normal distribution
                mu_x_sq = cond**2
                std_x = cond * change_scale
                var_x = std_x**2
                mu = np.log((mu_x_sq) / np.sqrt(var_x + mu_x_sq))
                sigma = np.sqrt(np.log(var_x / (mu_x_sq) + 1))

                z = jax.random.normal(subkey, shape=(n_signals - 1,))
                sampled_values = jnp.exp(mu + sigma * z)

                initial_conditions_ = initial_conditions_.at[1:, i].set(sampled_values)
    else:
        initial_conditions_ = jnp.tile(initial_conditions, (n_signals, 1))

    # save sampled parameters and initial conditions
    np.savetxt(
        os.path.join(experiment_folder, "sampled_parameters.csv"),
        sampled_params,
        delimiter=",",
        fmt="%.6f",
    )
    np.savetxt(
        os.path.join(experiment_folder, "sampled_initial_conditions.csv"),
        initial_conditions_,
        delimiter=",",
        fmt="%.6f",
    )

    return sampled_params, initial_conditions_


def sample_within_intervals(
    min_params,
    max_params,
    min_initial_conditions,
    max_initial_conditions,
    n_samples,
    key,
):
    """
    Sample model parameters and initial conditions uniformly within specified intervals.
    Args:
        min_params: Array of minimum values for model parameters.
        max_params: Array of maximum values for model parameters.
        min_initial_conditions: Array of minimum values for initial conditions.
        max_initial_conditions: Array of maximum values for initial conditions.
        n_samples: Number of samples to generate.
        key: JAX random key for sampling.
    Returns:
        sampled_params: Array of shape (n_samples, n_params) with sampled model parameters.
        sampled_initial_conditions: Array of shape (n_samples, n_initial_conditions) with sampled initial conditions.
    """
    n_p = min_params.shape[0]
    n_ic = min_initial_conditions.shape[0]
    sampled_params = jnp.zeros((n_samples, n_p))
    sampled_initial_conditions = jnp.zeros((n_samples, n_ic))

    keys = jax.random.split(key, n_p + n_ic)

    for i in range(n_p):
        sampled_params = sampled_params.at[:, i].set(
            jax.random.uniform(
                keys[i],
                shape=(n_samples,),
                minval=min_params[i],
                maxval=max_params[i],
            )
        )

    for j in range(n_ic):
        sampled_initial_conditions = sampled_initial_conditions.at[:, j].set(
            jax.random.uniform(
                keys[n_p + j],
                shape=(n_samples,),
                minval=min_initial_conditions[j],
                maxval=max_initial_conditions[j],
            )
        )

    return sampled_params, sampled_initial_conditions


##### INITIALIZATION OF PARAMETERS AND INITIAL CONDITIONS FOR UDE TRAINING #####


def init_with_known_values(config, project_root, experiment_folder, n_samples, key=None):
    """
    Initialize model parameters and initial conditions with known values from CSV files.
    Args:
        config: Configuration dictionary containing paths to parameter files.
        project_root: Root directory path for the project.
        experiment_folder: Folder path to save known parameters and initial conditions.
        n_samples: Number of samples.
        key: JAX random key (unused here but included for consistency).
    Returns:
        model_params: Array of model parameters loaded from CSV file.
        initial_conditions: Array of initial conditions loaded from CSV file.
        min_p: Minimum values of model parameters with buffer.
        max_p: Maximum values of model parameters with buffer.
        min_ic: Minimum values of initial conditions with buffer.
        max_ic: Maximum values of initial conditions with buffer.
    """
    model_params_path, initial_conditions_path = (
        config["model_parameters_path"],
        config["initial_conditions_path"],
    )
    model_params = jnp.array(
        pd.read_csv(
            Path(project_root) / model_params_path, header=None, index_col=False
        ).values
    )
    initial_conditions = jnp.array(
        pd.read_csv(
            Path(project_root) / initial_conditions_path, header=None, index_col=False
        ).values
    )

    np.savetxt(
        os.path.join(experiment_folder, "known_parameters.csv"),
        model_params,
        delimiter=",",
        fmt="%.6f",
    )
    np.savetxt(
        os.path.join(experiment_folder, "known_initial_conditions.csv"),
        initial_conditions,
        delimiter=",",
        fmt="%.6f",
    )

    buffer = 0.9  # 90% buffer to prevent boundary issues
    min_p = jnp.min(model_params, axis=0) * (1 - buffer)
    max_p = jnp.max(model_params, axis=0) * (1 + buffer)
    min_ic = jnp.min(initial_conditions, axis=0) * (1 - buffer)
    max_ic = jnp.max(initial_conditions, axis=0) * (1 + buffer)

    return model_params, initial_conditions, min_p, max_p, min_ic, max_ic


def init_with_resampling(config, project_root, experiment_folder, n_samples, key):
    """
    Initialize model parameters and initial conditions by resampling around known values.
    Args:
        config: Configuration dictionary containing known parameter values.
        project_root: Root directory path for the project.
        experiment_folder: Folder path to save resampled parameters and initial conditions.
        n_samples: Number of samples to generate.
        key: JAX random key for sampling.
    Returns:
        model_params: Array of resampled model parameters.
        initial_conditions: Array of resampled initial conditions.
        min_p: Minimum values of resampled model parameters with buffer.
        max_p: Maximum values of resampled model parameters with buffer.
        min_ic: Minimum values of resampled initial conditions with buffer.
        max_ic: Maximum values of resampled initial conditions with buffer.
    """
    model_params, initial_conditions, change_scale = (
        jnp.array(config["model_parameters"]),
        jnp.array(config["initial_conditions"]),
        config["change_scale"],
    )
    resampled_params, resampled_initial_conditions = random_sample_parameters(
        model_params,
        initial_conditions,
        n_samples,
        change_scale,
        True,
        experiment_folder,
        key,
    )

    buffer = 0.9  # 90% buffer to prevent boundary issues
    min_p = jnp.min(resampled_params, axis=0) * (1 - buffer)
    max_p = jnp.max(resampled_params, axis=0) * (1 + buffer)
    min_ic = jnp.min(resampled_initial_conditions, axis=0) * (1 - buffer)
    max_ic = jnp.max(resampled_initial_conditions, axis=0) * (1 + buffer)

    return resampled_params, resampled_initial_conditions, min_p, max_p, min_ic, max_ic


def init_within_intervals(config, project_root, experiment_folder, n_samples, key):
    """
    Initialize model parameters and initial conditions by sampling uniformly within specified intervals.
    Args:
        config: Configuration dictionary containing min and max parameter values.
        project_root: Root directory path for the project.
        experiment_folder: Folder path to save sampled parameters and initial conditions.
        n_samples: Number of samples to generate.
        key: JAX random key for sampling.
    Returns:
        model_params: Array of sampled model parameters.
        initial_conditions: Array of sampled initial conditions.
        min_p: Minimum values of model parameters.
        max_p: Maximum values of model parameters.
        min_ic: Minimum values of initial conditions.
        max_ic: Maximum values of initial conditions.
    """
    min_params = jnp.array(config["min_parameters"])
    max_params = jnp.array(config["max_parameters"])
    min_initial_conditions = jnp.array(config["min_initial_conditions"])
    max_initial_conditions = jnp.array(config["max_initial_conditions"])

    model_params, initial_conditions = sample_within_intervals(
        min_params,
        max_params,
        min_initial_conditions,
        max_initial_conditions,
        n_samples,
        key,
    )

    np.savetxt(
        os.path.join(experiment_folder, "parameters_sampled_within_intervals.csv"),
        model_params,
        delimiter=",",
        fmt="%.6f",
    )
    np.savetxt(
        os.path.join(
            experiment_folder, "initial_conditions_sampled_within_intervals.csv"
        ),
        initial_conditions,
        delimiter=",",
        fmt="%.6f",
    )

    return (
        model_params,
        initial_conditions,
        min_params,
        max_params,
        min_initial_conditions,
        max_initial_conditions,
    )


initialize_for_ude = {
    "init_known": init_with_known_values,
    "init_resample": init_with_resampling,
    "init_interval": init_within_intervals,
}
