"""Single-phase (incompressible or mildly compressible) pressure gradient."""

import warnings
from typing import NamedTuple

import numpy as np
import scipy.constants as SPC

from .dimensionless import reynolds
from .friction import friction_factor


class GradientResult(NamedTuple):
    """Pressure gradient decomposition [Pa/m], flow-direction sign convention.

    Supports tuple unpacking: ``total, dpg, dpf, dpv = result``.
    """

    total: float
    gravity: float
    friction: float
    momentum: float


def single_phase_gradient(
    mass_rate,
    D,
    density,
    viscosity,
    *,
    inclination=0.0,
    roughness=1.5e-4,
    compressibility=0.0,
) -> GradientResult:
    """Adiabatic single-phase pressure gradient in a constant-diameter pipe.

    Parameters
    ----------
    mass_rate : float or array_like
        Mass rate [kg/s]. Negative values mean reversed flow.
    D : float
        Pipe diameter [m].
    density : float or array_like
        Fluid density [kg/m3].
    viscosity : float or array_like
        Dynamic viscosity [Pa.s].
    inclination : float, optional
        ``sin(angle)``, in ``[-1, 1]``. Default 0 (horizontal).
    roughness : float, optional
        Absolute roughness [m]. Default 0.15 mm.
    compressibility : float, optional
        Fluid compressibility [1/Pa]. Default 0 (incompressible).

    Returns
    -------
    GradientResult
        ``(total, gravity, friction, momentum)`` in Pa/m. Negative values are
        losses in the direction of positive flow (see package docstring).

    Notes
    -----
    The momentum term assumes density change is pressure-driven:
    ``d(rho)/dx = d(rho)/dP * dP/dx``.
    """
    if not (np.all(inclination >= -1) and np.all(inclination <= 1)):
        raise ValueError("inclination must satisfy -1 <= inc <= 1")

    dpg = -SPC.g * inclination * density

    area = np.pi * D**2 / 4
    v = np.abs(mass_rate) / (density * area)
    re = reynolds(v, D, density, viscosity)
    f = friction_factor(re, D=D, eps=roughness, fanning=False)
    dpf = -f / (2 * D) * density * v**2 * np.sign(mass_rate)

    # momentum (kinetic) correction
    eh = compressibility * density * v**2
    if np.any(eh >= 1):
        raise ValueError("Supersonic flow encountered")
    if np.any(eh > 0.9):
        warnings.warn("Flow is close to supersonic", stacklevel=2)
    dpv = (dpf + dpg) * eh / (1 - eh)

    return GradientResult(dpg + dpf + dpv, dpg, dpf, dpv)
