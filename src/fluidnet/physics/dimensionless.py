"""Dimensionless numbers. Pure SI functions, numpy-vectorized."""

import numpy as np
import scipy.constants as spc

from fluidnet._types import ArrayLike


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
        density: ArrayLike,
        compressibility: ArrayLike,
        ) -> ArrayLike:
    """Mach number ``M = v / c`` with ``c = 1 / sqrt(rho * beta)``.

    Parameters
    ----------
    velocity : ArrayLike
        Flow velocity [m/s].
    density : ArrayLike
        Density [kg/m3].
    compressibility : ArrayLike
        Compressibility ``beta = (1/rho) * (d rho / d P)`` [1/Pa].

    Returns
    -------
    ArrayLike
        Mach number (dimensionless).

    Notes
    -----
    The speed of sound is obtained from the compressibility, 
    ``c**2 = 1 / (rho * beta)``, rather than from an ideal-gas
    expression. No equation of state is assumed here.

    The thermodynamic process is encoded in *which* ``beta`` the caller
    provides: an isothermal ``beta`` yields the isothermal Mach number, an
    isentropic one yields the acoustic Mach number. The two differ by a
    factor of ``sqrt(gamma)``.

    ``M**2 = rho * beta * v**2`` is the kinetic correction term appearing in
    the denominator of the pressure gradient; ``M = 1`` is where that
    gradient becomes singular (choked flow).
    """
    return velocity * np.sqrt(density * compressibility)