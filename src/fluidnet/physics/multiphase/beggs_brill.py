"""Beggs & Brill correlation for two-phase (liquid-gas) flow.

Public API: :func:`beggs_brill_gradient` -> :class:`GradientResult`.
Intermediate quantities (regime, holdup, NFr, friction factors) live in the
internal :func:`_beggs_brill_detailed`, used by golden tests and — later — by
the ``@diagnostic`` side channel. They are deliberately *not* part of the
public signature (no ``full_output`` flag).
"""

import warnings
from typing import Any, cast

import numpy as np
import scipy.constants as spc

from fluidnet.physics.dimensionless import froude, reynolds
from fluidnet.physics.friction import friction_factor
from fluidnet.physics.types import ArrayLike, GradientResult

FLOW_REGIMES = ("segregated", "intermittent", "distributed", "transition")


def beggs_brill_flowmap(Cl: ArrayLike, NFr: ArrayLike) -> int | np.ndarray:
    """Flow regime index from no-slip liquid fraction and Froude**2 number.

    Vectorized (uses boolean masks, no ``if``).

    Parameters
    ----------
    Cl : ArrayLike
        No-slip liquid fraction ``ql / (ql + qg)``, in ``[0, 1]``.
    NFr : ArrayLike
        Froude number squared, ``Fr**2 = v_mix**2 / (g * D)``.

    Returns
    -------
    int or ndarray
        Index into :data:`FLOW_REGIMES` for each element.

    Raises
    ------
    ValueError
        If some ``(Cl, NFr)`` pair does not fall into any regime boundary
        (e.g. NaN input).
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
        raise ValueError(f"invalid values for Cl {np.atleast_1d(Cl)[bad]} "
                         "and NFr {np.atleast_1d(NFr)[bad]}")
    regime: int | np.ndarray = m1 * 1 + m2 * 2 + m3 * 3
    return regime


def _holdup(
    i: int,
    Cl: ArrayLike,
    NFr: ArrayLike,
    Nlv: ArrayLike,
    angle: float,
) -> ArrayLike:
    """Liquid holdup for regime index ``i`` at pipe ``angle`` [rad].

    Parameters
    ----------
    i : int
        Flow regime index into :data:`FLOW_REGIMES` (0-3).
    Cl : ArrayLike
        No-slip liquid fraction.
    NFr : ArrayLike
        Froude number squared.
    Nlv : ArrayLike
        Liquid velocity number.
    angle : float
        Pipe inclination angle [rad], positive uphill.

    Returns
    -------
    ArrayLike
        Liquid holdup, not yet clipped to ``[0, 1]``.
    """
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
    *,
    liquid_mass_rate: float,
    gas_mass_rate: float,
    rho_liquid: float,
    rho_gas: float,
    mu_liquid: float,
    mu_gas: float,
    D: float,
    inclination: float = 0.0,
    roughness: float = 1.5e-4,
    sigma: float = 30.0e-3,
    compressibility: float = 0.0,
    holdup_adj: float = 1.0,
    payne_correction: bool = True,
) -> dict[str, Any]:
    """Full Beggs & Brill calculation, returning gradient plus intermediates.

    Internal: consumed by golden tests and the future diagnostics channel.
    Same sign convention as the rest of the package. Scalar-only contract
    today (see module docstring); ``beggs_brill_gradient`` is the public
    entry point built on top of this.

    Parameters
    ----------
    liquid_mass_rate, gas_mass_rate : float
        Mass rates [kg/s]. Both non-negative; flow direction is resolved
        by the integrator, not here. Negative rates raise ``ValueError``.
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
        Surface tension [N/m]. Default 30e-3.
    compressibility : float, optional
        Mixture compressibility [1/Pa] for the momentum term. Default 0.
    holdup_adj : float, optional
        Holdup multiplier (result clipped to [0, 1]). Default 1.
    payne_correction : bool, optional
        Apply Payne et al. holdup correction. Default True.

    Returns
    -------
    dict[str, Any]
        ``gradient`` (:class:`GradientResult`) plus intermediates:
        ``flow_regime``, ``NFr``, ``liquid_fraction``, ``liquid_holdup``,
        ``mixture_density``, ``liquid_velocity_number``, ``ReNs``, ``fNs``,
        ``f``.
    """
    grad = np.zeros(3)

    if liquid_mass_rate < 0 or gas_mass_rate < 0:
        raise ValueError(
            f"negative rates not allowed (ql={liquid_mass_rate:.3f} kg/s, "
            f"qg={gas_mass_rate:.3f} kg/s); flow direction is resolved by "
            f"the integrator, not physics"
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
    Nlv = vsl * (rho_liquid / (sigma * spc.g)) ** 0.25

    i = int(beggs_brill_flowmap(np.asarray(Cl), np.asarray(NFr)))

    el = _holdup(i, Cl, NFr, Nlv, np.arcsin(inclination))
    if payne_correction:
        el *= 0.924 if inclination > 0 else 0.685
    el = np.clip(el * holdup_adj, 0, 1)

    # gravity
    rho_mix = rho_liquid * el + rho_gas * (1 - el)
    grad[0] = -rho_mix * inclination * spc.g

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
    grad[1] = -2 * f / D * v_mix**2 * rho_ns

    # momentum
    eh = compressibility * rho_mix * v_mix**2
    if np.any(eh >= 1):
        raise ValueError("Supersonic flow encountered")
    if np.any(eh > 0.9):
        warnings.warn("Flow is close to supersonic", stacklevel=2)
    grad[2] = (grad[0] + grad[1]) * eh / (1 - eh)

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
    *,
    liquid_mass_rate: float,
    gas_mass_rate: float,
    rho_liquid: float,
    rho_gas: float,
    mu_liquid: float,
    mu_gas: float,
    D: float,
    inclination: float = 0.0,
    roughness: float = 1.5e-4,
    sigma: float = 30.0e-3,
    compressibility: float = 0.0,
    holdup_adj: float = 1.0,
    payne_correction: bool = True,
) -> GradientResult:
    """Beggs & Brill two-phase pressure gradient.

    Parameters
    ----------
    liquid_mass_rate, gas_mass_rate : float
        Mass rates [kg/s]. Both non-negative; flow direction is resolved
        by the integrator, not here. Negative rates raise ``ValueError``.
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
        Surface tension [N/m]. Default 30e-3.
    compressibility : float, optional
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
    grad = _beggs_brill_detailed(
        liquid_mass_rate=liquid_mass_rate,
        gas_mass_rate=gas_mass_rate,
        rho_liquid=rho_liquid,
        rho_gas=rho_gas,
        mu_liquid=mu_liquid,
        mu_gas=mu_gas,
        D=D,
        inclination=inclination,
        roughness=roughness,
        sigma=sigma,
        compressibility=compressibility,
        holdup_adj=holdup_adj,
        payne_correction=payne_correction,
    )["gradient"]
    return cast(GradientResult, grad)
