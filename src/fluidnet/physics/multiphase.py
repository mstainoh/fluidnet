"""Beggs & Brill correlation for two-phase (liquid-gas) flow.

Public API: :func:`beggs_brill_gradient` -> :class:`GradientResult`.
Intermediate quantities (regime, holdup, NFr, friction factors) live in the
internal :func:`_beggs_brill_detailed`, used by golden tests and — later — by
the ``@diagnostic`` side channel. They are deliberately *not* part of the
public signature (no ``full_output`` flag).
"""

import warnings

import numpy as np
import scipy.constants as SPC

from .dimensionless import froude, reynolds
from .friction import friction_factor
from .single_phase import GradientResult

FLOW_REGIMES = ("segregated", "intermittent", "distributed", "transition")


def beggs_brill_flowmap(Cl, NFr):
    """Flow regime index from no-slip liquid fraction and Froude**2 number.

    Returns an index into :data:`FLOW_REGIMES`. Vectorized.
    """
    L1 = 316 * Cl**0.302
    L2 = 0.0009252 * Cl**-2.4684
    L3 = 0.1 * Cl**-1.4516
    L4 = 0.5 * Cl**-6.738

    m0 = ((Cl <= 0.01) & (NFr <= L1)) | ((Cl > 0.01) & (NFr <= L2))
    m1 = ((Cl > 0.01) & (Cl <= 0.4) & (NFr > L3) & (NFr <= L1)) | (
        (Cl > 0.4) & (NFr > L3) & (NFr <= L4)
    )
    m2 = ((Cl <= 0.4) & (NFr > L1)) | ((Cl > 0.4) & (NFr > L4))
    m3 = (Cl > 0.01) & (NFr > L2) & (NFr <= L3)

    if not np.all(m0 | m1 | m2 | m3):
        bad = ~(m0 | m1 | m2 | m3)
        raise ValueError(f"invalid values for Cl {Cl[bad]} and NFr {NFr[bad]}")
    return m1 * 1 + m2 * 2 + m3 * 3


def _holdup(i, Cl, NFr, Nlv, angle):
    """Liquid holdup for regime index ``i`` at pipe ``angle`` [rad]."""
    if i == 0:
        a, b, c = 0.98, 0.4846, 0.0868  # segregated
    elif i == 1:
        a, b, c = 0.845, 0.5351, 0.0173  # intermittent
    elif i == 2:
        a, b, c = 1.065, 0.5824, 0.0609  # distributed
    elif i == 3:  # transition: interpolate intermittent/distributed
        L2 = 0.0009252 * Cl**-2.4684
        L3 = 0.1 * Cl**-1.4516
        A = (L3 - NFr) / (L3 - L2)
        return A * _holdup(1, Cl, NFr, Nlv, angle) + (1 - A) * _holdup(
            2, Cl, NFr, Nlv, angle
        )
    else:
        raise ValueError("regime index must be in 0..3")
    el0 = a * Cl**b / NFr**c

    if angle > 0 and i == 2:
        b_theta = 1.0
    else:
        if angle <= 0:
            d, e, f, g = 4.7, -0.3692, 0.1244, -0.5056
        elif i == 0:
            d, e, f, g = 0.011, -3.768, 3.539, -1.614
        else:  # i == 1
            d, e, f, g = 2.96, 0.305, -0.4473, 0.0978
        C = (1 - Cl) * np.log(d * Cl**e * Nlv**f * NFr**g)
        b_theta = 1 + np.clip(C, 0, np.inf) * (
            np.sin(1.8 * angle) - np.sin(1.8 * angle) ** 3 / 3
        )
    return el0 * b_theta


def _beggs_brill_detailed(
    liquid_mass_rate,
    gas_mass_rate,
    rho_liquid,
    rho_gas,
    mu_liquid,
    mu_gas,
    D,
    *,
    inclination=0.0,
    roughness=1.5e-4,
    sigma=30.0,
    mix_compressibility=0.0,
    holdup_adj=1.0,
    payne_correction=True,
) -> dict:
    """Full Beggs & Brill calculation, returning gradient plus intermediates.

    Internal: consumed by golden tests and the future diagnostics channel.
    Same sign convention as the rest of the package.
    """
    grad = np.zeros(4)

    if liquid_mass_rate > 0 and gas_mass_rate >= 0:
        inverse_flow = False
    elif liquid_mass_rate <= 0 and gas_mass_rate <= 0:
        inverse_flow = True
        liquid_mass_rate = abs(liquid_mass_rate)
        gas_mass_rate = abs(gas_mass_rate)
        inclination = -inclination
    else:
        raise ValueError(
            f"counterflow not allowed (ql={liquid_mass_rate:.3f} kg/s, "
            f"qg={gas_mass_rate:.3f} kg/s)"
        )

    ql = liquid_mass_rate / rho_liquid
    qg = gas_mass_rate / rho_gas
    Cl = ql / (ql + qg)

    area = np.pi * D**2 / 4
    v_mix = (ql + qg) / area

    mu_ns = Cl * mu_liquid + (1 - Cl) * mu_gas
    rho_ns = Cl * rho_liquid + (1 - Cl) * rho_gas

    NFr = froude(v_mix, D) ** 2
    re_ns = reynolds(v_mix, D, rho_ns, mu_ns)
    vsl = ql / area
    Nlv = vsl * (rho_liquid / (0.001 * sigma * SPC.g)) ** 0.25

    i = int(beggs_brill_flowmap(np.asarray(Cl), np.asarray(NFr)))

    el = _holdup(i, Cl, NFr, Nlv, np.arcsin(inclination))
    if payne_correction:
        el *= 0.924 if inclination > 0 else 0.685
    el = np.clip(el * holdup_adj, 0, 1)

    # gravity
    rho_mix = rho_liquid * el + rho_gas * (1 - el)
    grad[1] = -rho_mix * inclination * SPC.g

    # friction (two-phase multiplier over no-slip Fanning factor)
    f_ns = friction_factor(re_ns, D=D, eps=roughness, fanning=True)
    if el == 0:
        f = f_ns
    else:
        y = Cl / el**2
        if 1 <= y < 1.2:
            s = np.log(2.2 * y - 1.2)
        else:
            ly = np.log(y)
            s = ly / (-0.0523 + 3.182 * ly - 0.8725 * ly**2 + 0.01853 * ly**4)
        f = f_ns * np.exp(s)
    grad[2] = -2 * f / D * v_mix**2 * rho_ns

    # momentum
    eh = mix_compressibility * rho_mix * v_mix**2
    if np.any(eh >= 1):
        raise ValueError("Supersonic flow encountered")
    if np.any(eh > 0.9):
        warnings.warn("Flow is close to supersonic", stacklevel=2)
    grad[3] = (grad[1] + grad[2]) * eh / (1 - eh)

    if inverse_flow:
        grad = -grad
    grad[0] = grad[1:].sum()

    return {
        "gradient": GradientResult(*grad),
        "flow_regime": FLOW_REGIMES[i],
        "NFr": NFr,
        "liquid_fraction": Cl,
        "liquid_holdup": el,
        "mixture_density": rho_mix,
        "liquid_velocity_number": Nlv,
        "ReNs": re_ns,
        "fNs": f_ns,
        "f": f,
    }


def beggs_brill_gradient(
    liquid_mass_rate,
    gas_mass_rate,
    rho_liquid,
    rho_gas,
    mu_liquid,
    mu_gas,
    D,
    *,
    inclination=0.0,
    roughness=1.5e-4,
    sigma=30.0,
    mix_compressibility=0.0,
    holdup_adj=1.0,
    payne_correction=True,
) -> GradientResult:
    """Beggs & Brill two-phase pressure gradient.

    Parameters
    ----------
    liquid_mass_rate, gas_mass_rate : float
        Mass rates [kg/s]. Both non-negative (co-current) or both
        non-positive (reversed); counterflow raises ``ValueError``.
    rho_liquid, rho_gas : float
        Phase densities [kg/m3].
    mu_liquid, mu_gas : float
        Phase viscosities [Pa.s].
    D : float
        Pipe diameter [m].
    inclination : float, optional
        ``sin(angle)``. Default 0 (horizontal).
    roughness : float, optional
        Absolute roughness [m]. Default 0.15 mm.
    sigma : float, optional
        Surface tension [dyn/cm]. Default 30.
    mix_compressibility : float, optional
        Mixture compressibility [1/Pa] for the momentum term. Default 0.
    holdup_adj : float, optional
        Holdup multiplier (result clipped to [0, 1]). Default 1.
    payne_correction : bool, optional
        Apply Payne et al. holdup correction. Default True.

    Returns
    -------
    GradientResult
        ``(total, gravity, friction, momentum)`` in Pa/m, flow-direction
        sign convention.
    """
    return _beggs_brill_detailed(
        liquid_mass_rate,
        gas_mass_rate,
        rho_liquid,
        rho_gas,
        mu_liquid,
        mu_gas,
        D,
        inclination=inclination,
        roughness=roughness,
        sigma=sigma,
        mix_compressibility=mix_compressibility,
        holdup_adj=holdup_adj,
        payne_correction=payne_correction,
    )["gradient"]
