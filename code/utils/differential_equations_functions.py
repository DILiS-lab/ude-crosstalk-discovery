import diffrax
import jax
import jax.numpy as jnp
from lineax import AutoLinearSolver
from optimistix import Chord
import equinox as eqx
from utils.assumed_crosstalk_functions import get_assumed_crosstalk_function


def ode_solve_in_parallel(
    model, t_steps, max_steps, dt0, rtol=1e-6, atol=1e-10, stiff_solver=False
):
    """
    Create a parallelized ODE solver function.
    Args:
        model: The ODE system to solve.
        t_steps: Array of time points at which to save the solution.
        max_steps: Maximum number of integration steps allowed.
        dt0: Initial time step size.
        rtol: Relative tolerance for the solver.
        atol: Absolute tolerance for the solver.
        stiff_solver: If True, use Kvaerno5 solver for stiff equations;
                      if False, use Tsit5 solver.
    Returns:
        A function that solves the ODE system in parallel over batches of initial
        conditions and parameters which itself returns solution trajectories.
    """
    if stiff_solver:
        linear_solver = AutoLinearSolver(well_posed=False)
        root_finder = Chord(rtol=rtol, atol=atol, linear_solver=linear_solver)
        solver = diffrax.Kvaerno5(root_finder=root_finder)
    else:
        solver = diffrax.Tsit5()

    def solve_one(y0_, args_):
        return diffrax.diffeqsolve(
            terms=diffrax.ODETerm(model),
            solver=solver,
            t0=t_steps[0],
            t1=t_steps[-1],
            dt0=dt0,
            y0=y0_,
            args=args_,
            saveat=diffrax.SaveAt(ts=t_steps),
            stepsize_controller=diffrax.PIDController(rtol=rtol, atol=atol),
            max_steps=max_steps,
        ).ys

    vmapped = eqx.filter_jit(jax.vmap(solve_one, in_axes=(0, 0), out_axes=1))

    def solve_in_batches(y0, params, batch_size=None):
        if batch_size is None:
            return vmapped(y0, params)

        N = y0.shape[0]
        outs = []
        for i in range(0, N, batch_size):
            y0_b = y0[i : i + batch_size]
            params_b = jax.tree.map(lambda x: x[i : i + batch_size], params)
            outs.append(vmapped(y0_b, params_b))
        return jnp.concatenate(outs, axis=1)

    return solve_in_batches


def ode_solve_in_parallel_with_nfkb(
    model, t_steps, max_steps, func_type, dt0, rtol=1e-6, atol=1e-10, stiff_solver=False
):
    """
    Create a parallelized ODE solver function that incorporates NF-κB signal crosstalk.
    Args:
        model: The ODE system to solve.
        t_steps: Array of time points at which to save the solution.
        max_steps: Maximum number of integration steps allowed.
        func_type: Type of crosstalk function to use ("sigmoid", "decreasing", "oscillatory").
        dt0: Initial time step size.
        rtol: Relative tolerance for the solver.
        atol: Absolute tolerance for the solver.
        stiff_solver: If True, use Kvaerno5 solver for stiff equations;
                      if False, use Tsit5 solver.
    Returns:
        A function that solves the ODE system in parallel over batches of initial
        conditions, parameters, and NF-κB signals which itself returns solution trajectories.
    """
    if stiff_solver:
        linear_solver = AutoLinearSolver(well_posed=False)
        root_finder = Chord(rtol=rtol, atol=atol, linear_solver=linear_solver)
        solver = diffrax.Kvaerno5(root_finder=root_finder)
    else:
        solver = diffrax.Tsit5()

    def solve_one_with_func(y0_, bio_params_, nfkb_signal_):
        crosstalk_func = get_assumed_crosstalk_function(
            t_steps, nfkb_signal_, func_type
        )
        args_with_func = (*bio_params_, crosstalk_func)

        return diffrax.diffeqsolve(
            terms=diffrax.ODETerm(model),
            solver=solver,
            t0=t_steps[0],
            t1=t_steps[-1],
            dt0=dt0,
            y0=y0_,
            args=args_with_func,
            saveat=diffrax.SaveAt(ts=t_steps),
            stepsize_controller=diffrax.PIDController(rtol=rtol, atol=atol),
            max_steps=max_steps,
        ).ys

    vmapped_solve = eqx.filter_jit(
        jax.vmap(solve_one_with_func, in_axes=(0, 0, 0), out_axes=1)
    )

    def solve_with_nfkb(y0, params_and_signals, batch_size=None):
        model_params, nfkb_signals = params_and_signals
        n_samples = y0.shape[0]

        if batch_size is None:
            return vmapped_solve(y0, model_params, nfkb_signals.T)

        all_solutions = []
        for i in range(0, n_samples, batch_size):
            batch_end = min(i + batch_size, n_samples)
            batch_indices = jnp.arange(i, batch_end)

            y0_batch = y0[batch_indices]
            params_batch = model_params[batch_indices]
            signals_batch = nfkb_signals[:, batch_indices]
            signals_batch_vmapped = signals_batch.T

            batch_solution = vmapped_solve(
                y0_batch, params_batch, signals_batch_vmapped
            )
            all_solutions.append(batch_solution)

        return jnp.concatenate(all_solutions, axis=1)

    return solve_with_nfkb


def ude_solve_in_parallel(
    model, t_steps, max_steps, dt0, rtol=1e-6, atol=1e-10, stiff_solver=False
):
    """
    Create a parallelized solver function for UDEs.
    Args:
        model: The baseline ODE system.
        t_steps: Array of time points at which to save the solution.
        max_steps: Maximum number of integration steps allowed.
        dt0: Initial time step size.
        rtol: Relative tolerance for the solver.
        atol: Absolute tolerance for the solver.
        stiff_solver: If True, use Kvaerno5 solver for stiff equations;
                      if False, use Tsit5 solver.
    Returns:
        A function that solves the UDE system in parallel over batches of initial
        conditions, parameters, and NF-κB signals which itself returns solution trajectories.
    """
    if stiff_solver:
        linear_solver = AutoLinearSolver(well_posed=False)
        root_finder = Chord(rtol=rtol, atol=atol, linear_solver=linear_solver)
        solver = diffrax.Kvaerno5(root_finder=root_finder)
        adjoint = diffrax.RecursiveCheckpointAdjoint()
    else:
        solver = diffrax.Tsit5()
        adjoint = diffrax.BacksolveAdjoint()

    def solve_one_with_func(y0_, bio_params_, nfkb_signal_, neural_net_):
        args_ = (*bio_params_, t_steps, nfkb_signal_, neural_net_)

        return diffrax.diffeqsolve(
            terms=diffrax.ODETerm(model),
            solver=solver,
            t0=t_steps[0],
            t1=t_steps[-1],
            dt0=dt0,
            y0=y0_,
            args=args_,
            saveat=diffrax.SaveAt(ts=t_steps),
            stepsize_controller=diffrax.PIDController(rtol=rtol, atol=atol),
            max_steps=max_steps,
            adjoint=adjoint,
        ).ys

    vmapped_solve = eqx.filter_jit(
        jax.vmap(solve_one_with_func, in_axes=(0, 0, 0, None), out_axes=1)
    )

    def solve(y0, params_and_signals, neural_net_, batch_size=None):
        model_params, nfkb_signals = params_and_signals
        n_samples = y0.shape[0]

        if batch_size is None:
            return vmapped_solve(y0, model_params, nfkb_signals.T, neural_net_)

        all_solutions = []
        for i in range(0, n_samples, batch_size):
            batch_end = min(i + batch_size, n_samples)
            batch_indices = jnp.arange(i, batch_end)

            y0_batch = y0[batch_indices]
            params_batch = model_params[batch_indices]
            signals_batch = nfkb_signals[:, batch_indices]
            signals_batch_vmapped = signals_batch.T

            batch_solution = vmapped_solve(
                y0_batch, params_batch, signals_batch_vmapped, neural_net_
            )
            all_solutions.append(batch_solution)

        return jnp.concatenate(all_solutions, axis=1)

    return solve
