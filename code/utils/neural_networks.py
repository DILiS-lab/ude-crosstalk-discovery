import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp


class SynthNN(eqx.Module):
    """
    A simple feedforward neural network model for synthetic data generation.
    This model consists of an MLP with one input node, one output node, and
    two hidden layers with 16 nodes each. The activation function used is ReLU.
    The output is transformed using the softplus function to ensure non-negativity.
    Attributes:
        mlp: An Equinox MLP model.
    """

    mlp: eqx.nn.MLP

    def __init__(self, key):
        self.mlp = eqx.nn.MLP(
            in_size=1, out_size=1, width_size=16, depth=2, activation=jnn.relu, key=key
        )

    def __call__(self, x):
        x = jnp.atleast_1d(x)
        y = self.mlp(x)
        # return 2 * jnn.sigmoid(jnp.squeeze(y))
        return jnn.softplus(jnp.squeeze(y))


class ConstantNN(eqx.Module):
    """
    A Neural Network that always returns 1.0.
    Used to simulate the 'No Crosstalk' condition within the UDE framework.
    """
    def __init__(self, key=None):
        # No parameters to initialize
        pass

    def __call__(self, x):
        # Return 1.0 regardless of input. 
        # Since p53_synthesis = sigma * NN(nfkb), this results in p53_synthesis = sigma (constant).
        return jnp.array(1.0)
