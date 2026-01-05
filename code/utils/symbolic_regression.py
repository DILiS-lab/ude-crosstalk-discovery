from pysr import PySRRegressor
import os
import jax.numpy as jnp


def make_value_weights(nfkb_values, bins=20, eps=1e-6):
    """
    Create weights inversely proportional to the density of nfkb_values.
    Args:
        nfkb_values: Array of NF-κB signal values.
        bins: Number of bins to use for histogram.
        eps: Small value to avoid division by zero.
    Returns:
        Weights array with the same shape as nfkb_values.
    """

    nfkb_values = jnp.asarray(nfkb_values).ravel()
    counts, edges = jnp.histogram(nfkb_values, bins=bins)
    bin_idx = jnp.clip(
        jnp.digitize(nfkb_values, edges[:-1]) - 1, 0, counts.shape[0] - 1
    )
    w = 1.0 / (counts[bin_idx] + eps)
    w = w / jnp.maximum(jnp.mean(w), 1e-12)
    return w


def run_symbolic_regression(experiment_folder, input, predictions):
    """
    Run symbolic regression to find an interpretable function mapping input to predictions.
    Args:
        experiment_folder: Folder path to save the symbolic regression log.
        input: Input data array (2D).
        predictions: Target predictions array (2D).
    Returns:
        A tuple containing:
            - function: The symbolic regression function as a JAX callable.
            - parameters: The parameters of the symbolic regression function.
    """

    model = PySRRegressor(
        niterations=1000,
        unary_operators=["exp", "log", "square", "sin", "cos", "tan"],
        maxsize=12,
        model_selection="best",
        temp_equation_file=True,
        verbosity=0,
        progress=False,
        update=False,
        batching=True,
    )

    weights = make_value_weights(input)

    model.fit(input, predictions, weights=weights)

    # find best equation and its score
    best_equation = model.get_best()
    print("Best equation found:", best_equation["equation"])
    print("Model score:", model.score(input, predictions))

    # save results to a file
    log_file = os.path.join(experiment_folder, "symbolic_regression_log.txt")
    with open(log_file, "a") as f:
        f.write(f"Best equation: {best_equation['equation']}\n")
        f.write(f"Model score: {model.score(input, predictions)}\n")

    # convert best equation to jax function for use in UDE
    model_jax = model.jax()
    function = model_jax["callable"]
    parameters = model_jax["parameters"]

    return function, parameters
