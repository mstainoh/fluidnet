"""Dimensionless numbers. Pure SI functions, numpy-vectorized."""

import numpy as np
import scipy.constants as SPC


def reynolds(v, D, density, viscosity):
    """Reynolds number ``Re = v * D * rho / mu``.

    Parameters
    ----------
    v : float or array_like
        Velocity [m/s].
    D : float or array_like
        Diameter [m].
    density : float or array_like
        Density [kg/m3].
    viscosity : float or array_like
        Dynamic viscosity [Pa.s].
    """
    return v * D * density / viscosity


def froude(v, D):
    """Froude number ``Fr = v / sqrt(g * D)``.

    Parameters
    ----------
    v : float or array_like
        Velocity [m/s].
    D : float or array_like
        Diameter [m].
    """
    return v / np.sqrt(SPC.g * D)


def mach(velocity, pressure, density, gamma=1.4):
    """Mach number ``M = v / c`` with ``c = sqrt(gamma * P / rho)``.

    Parameters
    ----------
    velocity : float or array_like
        Flow velocity [m/s].
    pressure : float or array_like
        Absolute pressure [Pa].
    density : float or array_like
        Density [kg/m3].
    gamma : float, optional
        Heat capacity ratio (default 1.4).
    """
    c = np.sqrt(gamma * pressure / density)
    return velocity / c
