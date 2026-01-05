from pathlib import Path
import pickle
import jax.numpy as jnp


def load_symbolic_model_externally(experiment_folder):
    """
    Load the symbolic regression model from the experiment folder.

    Args:
        experiment_folder: Path to the experiment folder containing the .pkl file.

    Returns:
        The symbolic regression model as a JAX function.
    """
    experiment_folder = Path(experiment_folder)
    model_path = experiment_folder / "symbolic_regression_model.pkl"

    print(f"Loading model from {model_path}")
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Recreate the JAX function and parameters from the PySR model
    model_jax = model.jax()
    symbolic_reg_function = model_jax["callable"]
    s_r_params = model_jax["parameters"]

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

    return symbolic_reg_model
