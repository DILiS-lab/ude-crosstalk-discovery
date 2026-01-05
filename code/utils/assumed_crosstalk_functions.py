import jax.numpy as jnp

func = {
    "sigmoid": lambda x: 2 / (1 + jnp.exp(-x)),
    "decreasing": lambda x: 1 + 2 / (x + 1),
    "oscillatory": lambda x: jnp.abs(jnp.sin(x)),
}


def get_assumed_crosstalk_function(t_obs, nfkb_signal, type):
    """
    Generate a crosstalk function that models p53 synthesis as a function of NF-κB signal.
    This factory function creates a closure that interpolates NF-κB signal values at any time point
    and applies a type-specific transformation function to model p53 synthesis based on NF-κB levels.
    Args:
        t_obs: Observed time points.
        nfkb_signal: NF-κB signal values corresponding to each time point in t_obs.
        type: The type of transformation function to apply. One of: "sigmoid", "decreasing", "oscillatory".
    Returns:
        A function f_p53_synthesis_nfkb(t) that takes a time point t and returns
                 the p53 synthesis rate based on interpolated NF-κB signal at that time.
                 The function signature is f_p53_synthesis_nfkb(t).
    """

    def f_p53_synthesis_nfkb(t):
        nfkb = jnp.interp(t, t_obs, nfkb_signal)
        return func[type](nfkb)

    return f_p53_synthesis_nfkb
