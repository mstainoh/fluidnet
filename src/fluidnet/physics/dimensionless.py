"""Dimensionless numbers. Pure SI functions, numpy-vectorized."""

import numpy as np
import scipy.constants as spc

from .types import ArrayLike


def reynolds(
        v: ArrayLike,
        D: ArrayLike,
        density: ArrayLike,
        viscosity: ArrayLike,) -> ArrayLike:
    """Reynolds number ``Re = v * D * rho / mu``.

    Parameters
    ----------
    v : ArrayLike
        Velocity [m/s].
    D : ArrayLike
        Diameter [m].
    density : ArrayLike
        Density [kg/m3].
    viscosity : ArrayLike
        Dynamic viscosity [Pa.s].

    Returns
    -------
    ArrayLike
        Reynolds number (dimensionless).
    """
    return v * D * density / viscosity


def froude(v: ArrayLike, D: ArrayLike) -> ArrayLike:
    """Froude number ``Fr = v / sqrt(g * D)``.

    Parameters
    ----------
    v : ArrayLike
        Velocity [m/s].
    D : ArrayLike
        Diameter [m].

    Returns
    -------
    ArrayLike
        Froude number (dimensionless).
    """
    return v / np.sqrt(spc.g * D)


def mach(
        velocity: ArrayLike,
        pressure: ArrayLike,
        density: ArrayLike,
        gamma: float = 1.4
        ) -> ArrayLike:
    """Mach number ``M = v / c`` with ``c = sqrt(gamma * P / rho)``.

    Parameters
    ----------
    velocity : ArrayLike
        Flow velocity [m/s].
    pressure : ArrayLike
        Absolute pressure [Pa].
    density : ArrayLike
        Density [kg/m3].
    gamma : float, optional
        Heat capacity ratio (default 1.4).

    Returns
    -------
    ArrayLike
        Mach number (dimensionless).
    """
    c = np.sqrt(gamma * pressure / density)
    return velocity / c
