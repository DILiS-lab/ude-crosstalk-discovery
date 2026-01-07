import jax.numpy as jnp

####### NFKB SIGNAL GENERATION MODELS #######


def nfkb_signal_generation_zambrano(t, x, args):
    """
    NFKB signal generation model based on Zambrano et al.
    Args:
        t: Time point.
        x: State variables [K, N, I, R, G].
        args: Model parameters [d_k, d, gamma, d_l, p, a, k, k_p, d_r, k_on, k_off].
    Returns:
        Derivatives [dK/dt, dN/dt, dI/dt, dR/dt, dG/dt].
    """

    K, N, I, R, G = x
    d_k, d, gamma, d_l, p, a, k, k_p, d_r, k_on, k_off = args

    dx1 = -d_k * K
    dx2 = d * (1 - N) + gamma * d_l * (1 - N) + p * K * (1 - N) - a * N * I
    dx3 = d * (1 - N) - k * p * K * I - a * N * I + k_p * R - d_l * I
    dx4 = d_r * (G - R)
    dx5 = k_on * N * (1 - G) - k_off * I * G

    return jnp.array([dx1, dx2, dx3, dx4, dx5])


nfkb_signal_models = {
    "nfkb_zambrano": nfkb_signal_generation_zambrano,
}


####### P53 ODE MODELS WITHOUT NFKB #######


def ode_hunziker2010_without_nfkb(t, x, args):
    """
    p53 ODE model based on Hunziker et al. without NF-κB crosstalk.
    Args:
        t: Time point.
        x: State variables [p53, Mdm2mRNA, MDM2, MDM2p53].
        args: Model parameters [sigma_p, alpha, gamma, k_b, r, k_trs, beta, k_trd, delta].
    Returns:
        Derivatives [dp53/dt, dMdm2mRNA/dt, dMDM2/dt, dMDM2p53/dt].
    """

    p53, Mdm2mRNA, MDM2, MDM2p53 = x
    sigma_p, alpha, gamma, k_b, r, k_trs, beta, k_trd, delta = args
    k_f = k_b / r

    dx1 = sigma_p - alpha * p53 - k_f * p53 * MDM2 + gamma * MDM2p53 + k_b * MDM2p53
    dx2 = k_trs * (p53**2) - beta * Mdm2mRNA
    dx3 = (
        k_trd * Mdm2mRNA
        - k_f * MDM2 * p53
        + k_b * MDM2p53
        + delta * MDM2p53
        - gamma * MDM2
    )
    dx4 = k_f * MDM2 * p53 - k_b * MDM2p53 - delta * MDM2p53 - gamma * MDM2p53

    return jnp.array([dx1, dx2, dx3, dx4])


def ode_konrath2020_without_nfkb(t, x, args):
    """
    p53 ODE model based on Konrath et al. without NF-κB crosstalk.
    Args:
        t: Time point.
        x: State variables [p53, p53a, Mdm2_mRNA, Mdm2, Wip1_mRNA, Wip1, ATMP].
        args: Model parameters [damage, Ts, Tw, am, ampa, ampi, api, as_, asm, aw, awpa,
                               aws, bpamt, amt, bmt, bmtm, bp, bs, bsp, bpawt,
                               awt, bwt, bwtw, ns, nw, bmt_fc, bmtm_fc,
                               bp_fc, bs_fc, bwt_fc, bwtw_fc].
    Returns:
        Derivatives [dp53/dt, dp53a/dt, dMdm2_mRNA/dt, dMdm2/dt, dWip1_mRNA/dt,
                     dWip1/dt, dATMP/dt].
    """

    p53, p53a, Mdm2_mRNA, Mdm2, Wip1_mRNA, Wip1, ATMP = x
    (
        damage,
        Ts,
        Tw,
        am,
        ampa,
        ampi,
        api,
        as_,
        asm,
        aw,
        awpa,
        aws,
        bpamt,
        amt,
        bmt,
        bmtm,
        bp,
        bs,
        bsp,
        bpawt,
        awt,
        bwt,
        bwtw,
        ns,
        nw,
        bmt_fc,
        bmtm_fc,
        bp_fc,
        bs_fc,
        bwt_fc,
        bwtw_fc,
    ) = args

    dx1 = (
        bp * bp_fc
        + awpa * Wip1 * p53a
        - api * p53
        - ampi * Mdm2 * p53
        - bsp * p53 * ((ATMP**ns) / (ATMP**ns + Ts**ns))
    )
    dx2 = (
        bsp * p53 * ((ATMP**ns) / (ATMP**ns + Ts**ns))
        - ampa * Mdm2 * p53a
        - awpa * Wip1 * p53a
    )
    dx3 = bmt * bmt_fc + bpamt * p53a - amt * Mdm2_mRNA
    dx4 = bmtm * bmtm_fc * Mdm2_mRNA - asm * Mdm2 * ATMP - am * Mdm2
    dx5 = bwt * bwt_fc + bpawt * p53a - awt * Wip1_mRNA
    dx6 = bwtw * bwtw_fc * Wip1_mRNA - aw * Wip1
    dx7 = (
        bs * bs_fc * damage
        - aws * ATMP * ((Wip1**nw) / (Wip1**nw + Tw**nw))
        - as_ * ATMP
    )

    return jnp.array([dx1, dx2, dx3, dx4, dx5, dx6, dx7])


ode_models_p53_without_nfkb = {
    "hunziker2010": ode_hunziker2010_without_nfkb,
    "konrath2020": ode_konrath2020_without_nfkb,
}

####### P53 ODE MODELS WITH NFKB #######


def ode_hunziker2010_with_nfkb(t, x, args):
    """
    p53 ODE model based on Hunziker et al. with NF-κB crosstalk.
    Args:
        t: Time point.
        x: State variables [p53, Mdm2mRNA, MDM2, MDM2p53].
        args: Model parameters [sigma_p, alpha, gamma, k_b, r, k_trs, beta, k_trd, delta, function].
    Returns:
        Derivatives [dp53/dt, dMdm2mRNA/dt, dMDM2/dt, dMDM2p53/dt].
    """

    p53, Mdm2mRNA, MDM2, MDM2p53 = x
    sigma_p, alpha, gamma, k_b, r, k_trs, beta, k_trd, delta, function = args
    k_f = k_b / r
    p53_synth = function(t)

    dx1 = (
        sigma_p * p53_synth
        - alpha * p53
        - k_f * p53 * MDM2
        + gamma * MDM2p53
        + k_b * MDM2p53
    )
    dx2 = k_trs * (p53**2) - beta * Mdm2mRNA
    dx3 = (
        k_trd * Mdm2mRNA
        - k_f * MDM2 * p53
        + k_b * MDM2p53
        + delta * MDM2p53
        - gamma * MDM2
    )
    dx4 = k_f * MDM2 * p53 - k_b * MDM2p53 - delta * MDM2p53 - gamma * MDM2p53

    return jnp.array([dx1, dx2, dx3, dx4])


def ode_konrath2020_with_nfkb(t, x, args):
    """
    p53 ODE model based on Konrath et al. with NF-κB crosstalk.
    Args:
        t: Time point.
        x: State variables [p53, p53a, Mdm2_mRNA, Mdm2, Wip1_mRNA, Wip1, ATMP].
        args: Model parameters [damage, Ts, Tw, am, ampa, ampi, api, as_, asm, aw, awpa,
                               aws, bpamt, amt, bmt, bmtm, bp, bs, bsp, bpawt,
                               awt, bwt, bwtw, ns, nw, bmt_fc, bmtm_fc,
                               bp_fc, bs_fc, bwt_fc, bwtw_fc, function].
    Returns:
        Derivatives [dp53/dt, dp53a/dt, dMdm2_mRNA/dt, dMdm2/dt, dWip1_mRNA/dt,
                     dWip1/dt, dATMP/dt].
    """

    p53, p53a, Mdm2_mRNA, Mdm2, Wip1_mRNA, Wip1, ATMP = x
    (
        damage,
        Ts,
        Tw,
        am,
        ampa,
        ampi,
        api,
        as_,
        asm,
        aw,
        awpa,
        aws,
        bpamt,
        amt,
        bmt,
        bmtm,
        bp,
        bs,
        bsp,
        bpawt,
        awt,
        bwt,
        bwtw,
        ns,
        nw,
        bmt_fc,
        bmtm_fc,
        bp_fc,
        bs_fc,
        bwt_fc,
        bwtw_fc,
        function,
    ) = args
    p53_synth = function(t)

    dx1 = (
        bp * bp_fc * p53_synth
        + awpa * Wip1 * p53a
        - api * p53
        - ampi * Mdm2 * p53
        - bsp * p53 * ((ATMP**ns) / (ATMP**ns + Ts**ns))
    )
    dx2 = (
        bsp * p53 * ((ATMP**ns) / (ATMP**ns + Ts**ns))
        - ampa * Mdm2 * p53a
        - awpa * Wip1 * p53a
    )
    dx3 = bmt * bmt_fc + bpamt * p53a - amt * Mdm2_mRNA
    dx4 = bmtm * bmtm_fc * Mdm2_mRNA - asm * Mdm2 * ATMP - am * Mdm2
    dx5 = bwt * bwt_fc + bpawt * p53a - awt * Wip1_mRNA
    dx6 = bwtw * bwtw_fc * Wip1_mRNA - aw * Wip1
    dx7 = (
        bs * bs_fc * damage
        - aws * ATMP * ((Wip1**nw) / (Wip1**nw + Tw**nw))
        - as_ * ATMP
    )

    return jnp.array([dx1, dx2, dx3, dx4, dx5, dx6, dx7])


ode_models_p53_with_nfkb = {
    "hunziker2010": ode_hunziker2010_with_nfkb,
    "konrath2020": ode_konrath2020_with_nfkb,
}

######## UDE MODELS #######


def ude_hunziker2010(t, x, args):
    """
    UDE model based on Hunziker et al.
    Args:
        t: Time point.
        x: State variables [p53, Mdm2mRNA, MDM2, MDM2p53].
        args: Model parameters [sigma_p, alpha, gamma, k_b, r, k_trs, beta, k_trd, delta,
                              t_obs, nfkb_signal, neural_network].
    Returns:
        Derivatives [dp53/dt, dMdm2mRNA/dt, dMDM2/dt, dMDM2p53/dt].
    """

    x = jnp.clip(x, 0.0, 1e6)
    p53, Mdm2mRNA, MDM2, MDM2p53 = x
    (
        sigma_p,
        alpha,
        gamma,
        k_b,
        r,
        k_trs,
        beta,
        k_trd,
        delta,
        t_obs,
        nfkb_signal,
        neural_network,
    ) = args

    k_f = k_b / r

    nfkb_val = jnp.interp(t, t_obs, nfkb_signal)
    p53_synth = neural_network(nfkb_val)

    dx1 = (
        sigma_p * p53_synth
        - alpha * p53
        - k_f * p53 * MDM2
        + gamma * MDM2p53
        + k_b * MDM2p53
    )
    dx2 = k_trs * (p53**2) - beta * Mdm2mRNA
    dx3 = (
        k_trd * Mdm2mRNA
        - k_f * MDM2 * p53
        + k_b * MDM2p53
        + delta * MDM2p53
        - gamma * MDM2
    )
    dx4 = k_f * MDM2 * p53 - k_b * MDM2p53 - delta * MDM2p53 - gamma * MDM2p53

    return jnp.array([dx1, dx2, dx3, dx4])


def ude_konrath2020(t, x, args):
    """
    UDE model based on Konrath et al.
    Args:
        t: Time point.
        x: State variables [p53, p53a, Mdm2_mRNA, Mdm2, Wip1_mRNA, Wip1, ATMP].
        args: Model parameters [damage, Ts, Tw, am, ampa, ampi, api, as_, asm, aw, awpa,
                               aws, bpamt, amt, bmt, bmtm, bp, bs, bsp, bpawt,
                               awt, bwt, bwtw, ns, nw, bmt_fc, bmtm_fc,
                               bp_fc, bs_fc, bwt_fc, bwtw_fc,
                               t_obs, nfkb_signal, neural_network].
    Returns:
        Derivatives [dp53/dt, dp53a/dt, dMdm2_mRNA/dt, dMdm2/dt, dWip1_mRNA/dt,
                     dWip1/dt, dATMP/dt].
    """

    x = jnp.clip(x, 0.0, 1e6)

    (
        damage,
        Ts,
        Tw,
        am,
        ampa,
        ampi,
        api,
        as_,
        asm,
        aw,
        awpa,
        aws,
        bpamt,
        amt,
        bmt,
        bmtm,
        bp,
        bs,
        bsp,
        bpawt,
        awt,
        bwt,
        bwtw,
        ns,
        nw,
        bmt_fc,
        bmtm_fc,
        bp_fc,
        bs_fc,
        bwt_fc,
        bwtw_fc,
        t_obs,
        nfkb_signal,
        neural_network,
    ) = args
    p53, p53a, Mdm2_mRNA, Mdm2, Wip1_mRNA, Wip1, ATMP = x

    nfkb_val = jnp.interp(t, t_obs, nfkb_signal)
    p53_synth = neural_network(nfkb_val)

    dx1 = (
        bp * bp_fc * p53_synth
        + awpa * Wip1 * p53a
        - api * p53
        - ampi * Mdm2 * p53
        - bsp * p53 * ((ATMP**ns) / (ATMP**ns + Ts**ns))
    )
    dx2 = (
        bsp * p53 * ((ATMP**ns) / (ATMP**ns + Ts**ns))
        - ampa * Mdm2 * p53a
        - awpa * Wip1 * p53a
    )
    dx3 = bmt * bmt_fc + bpamt * p53a - amt * Mdm2_mRNA
    dx4 = bmtm * bmtm_fc * Mdm2_mRNA - asm * Mdm2 * ATMP - am * Mdm2
    dx5 = bwt * bwt_fc + bpawt * p53a - awt * Wip1_mRNA
    dx6 = bwtw * bwtw_fc * Wip1_mRNA - aw * Wip1
    dx7 = (
        bs * bs_fc * damage
        - aws * ATMP * ((Wip1**nw) / (Wip1**nw + Tw**nw))
        - as_ * ATMP
    )

    return jnp.array([dx1, dx2, dx3, dx4, dx5, dx6, dx7])


ude_models = {
    "hunziker2010": ude_hunziker2010,
    "konrath2020": ude_konrath2020,
}


######## TRANSFORMING PARAMETER BEFORE AND AFTER EQUILIBRATION #######


def transform_params_hunziker(params):
    """
    Transform parameters for Hunziker et al. model before and after equilibration.
    Args:
        params: Array of model parameters.
    Returns:
        Tuple of (params_before_equilibration, params_after_equilibration).
    """

    return params, params.at[:, [2, 4, 8]].set(
        params[:, [2, 4, 8]] * jnp.array([2.5, 10.0, 1 / 5.5])
    )


def transform_params_konrath(params):
    """
    Transform parameters for Konrath et al. model before and after equilibration.
    Args:
        params: Array of model parameters.
    Returns:
        Tuple of (params_before_equilibration, params_after_equilibration).
    """

    return params.at[:, 0].set(0.0), params.at[:, 0].set(1.0)


transform_params_functions = {
    "hunziker2010": transform_params_hunziker,
    "konrath2020": transform_params_konrath,
}

######## NEW PARAMETER BOUNDARIES #######


def new_boundaries_hunziker(min_params, max_params):
    """
    Set new parameter boundaries for Hunziker et al. model after equilibration.
    Args:
        min_params: Array of minimum parameter values.
        max_params: Array of maximum parameter values.
    Returns:
        Tuple of (new_min_params, new_max_params).
    """

    idx = jnp.array([2, 4])
    max_params = max_params.at[idx].set(max_params[idx] * jnp.array([2.5, 10.0]))
    min_params = min_params.at[8].set(min_params[8] * (1 / 5.5))

    return min_params, max_params


def new_boundaries_konrath(min_params, max_params):
    """
    Set new parameter boundaries for Konrath et al. model after equilibration.
    Args:
        min_params: Array of minimum parameter values.
        max_params: Array of maximum parameter values.
    Returns:
        Tuple of (new_min_params, new_max_params).
    """

    min_params = min_params.at[0].set(0.0)
    max_params = max_params.at[0].set(1.0)
    return min_params, max_params


new_boundaries_p = {
    "hunziker2010": new_boundaries_hunziker,
    "konrath2020": new_boundaries_konrath,
}


######## FINAL SOLUTION FORMATTING FUNCTIONS #######


def final_solution_format_hunziker(sol, os_=0.0, sc_=1.0):
    """
    Format final solution for Hunziker et al. model.
    Args:
        sol: Solution array with shape (time_points, samples, variables).
        os_: Offset value (default is 0.0).
        sc_: Scale value (default is 1.0).
    Returns:
        Formatted solution array with only p53 values.
    """
    return os_ + sc_ * sol[:, :, 0]


def final_solution_format_konrath(sol, os_=0.0, sc_=1.0):
    """
    Format final solution for Konrath et al. model.
    Args:
        sol: Solution array with shape (time_points, samples, variables).
        os_: Offset value (default is 0.0).
        sc_: Scale value (default is 1.0).
    Returns:
        Formatted solution array with combined p53 and p53a values.
    """
    return os_ + sc_ * (sol[:, :, 0] + sol[:, :, 1])  # p53 + p53a


final_solution_format = {
    "hunziker2010": final_solution_format_hunziker,
    "konrath2020": final_solution_format_konrath,
}
