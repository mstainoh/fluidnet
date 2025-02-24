# src/fluid_functions.py
"""
Fluid dynamics functions for calculating various fluid flow parameters.

This module includes functions to calculate Reynolds number, Froude number,
friction factor, and pressure gradient for single-phase fluid flow.

Functions
---------
reynolds(v, D, density=1000, viscosity=0.001)
    Calculate the Reynolds number.
froid(v, D)
    Calculate the Froude number.
_chen_approx(re, eD)
    Chen approximation for turbulent friction factor.
find_friction_factor(re, eD, fanning=True)
    Calculate the friction factor for fluid flow.
single_phase_pressure_gradient(flow_rate, D, density=1000, viscosity=1e-3, inc=0,
                               eps=0.15e-3, compressibility=0, L=1, K=0,
                               output_components=False, as_head=False)
    Calculate the pressure gradient for single-phase fluid flow.
"""

import numpy as np
from scipy import constants as spc
import functools
import warnings

def reynolds(v, D, density=1000, viscosity=0.001):
    """
    Calculate the Reynolds number.

    Parameters
    ----------
    v : float or array-like
        Velocity in m/s.
    D : float
        Diameter in m.
    density : float, optional
        Density in kg/m^3 (default: 1000).
    viscosity : float, optional
        Viscosity in Pa.s (default: 0.001).

    Returns
    -------
    float or array-like
        Reynolds number.
    """
    return v * D * density / viscosity


def froid(v, D):
    """
    Calculate the Froude number.

    Parameters
    ----------
    v : float or array-like
        Velocity in m/s.
    D : float
        Diameter in m.

    Returns
    -------
    float or array-like
        Froude number.
    """
    return v / np.sqrt(spc.g * D)


def _chen_approx(re, eD):
    """
    Chen approximation for turbulent friction factor.

    Parameters
    ----------
    re : float or array-like
        Reynolds number.
    eD : float
        Relative roughness (epsilon/D).

    Returns
    -------
    float or array-like
        Friction factor.
    """
    re = np.clip(re, 2e3, re)
    return (
        -4 * np.log10(
            0.2698 * eD - 5.0452 / re * np.log10(
                0.3539 * eD**1.1098 + 5.8506 / re**0.8981)
        )
    ) ** -2


def find_friction_factor(re, eD, fanning=True):
    """
    Calculate the friction factor for fluid flow.

    Parameters
    ----------
    re : float or array-like
        Reynolds number.
    eD : float
        Relative roughness (epsilon/D).
    fanning : bool, optional
        True to return Fanning friction factor, False for Darcy-Weisbach (default: True).

    Returns
    -------
    float or array-like
        Friction factor.
    """
    sgn = np.sign(re)
    re = np.abs(re)
    f = np.zeros_like(re, dtype=np.float64)
    m1 = (re > 1e-10) & (re <= 2000)
    m2 = (re > 2000) & (re < 4000)
    m3 = re >= 4000
    f[m1] = 16 / re[m1]
    f[m2 | m3] = _chen_approx(re[m2 | m3], eD)
    f[m2] = f[m2] * ((re[m2] - 2e3) + (16 / re[m2]) * (4e3 - re[m2])) / 2e3
    return f * (1 if fanning else 4) * sgn


def single_phase_pressure_gradient(
    flow_rate, *args, D=1, density=1000, viscosity=1e-3, inc=0,
    eps=0.15e-3, compressibility=0, L=1, K=0,
    output_components=False, full_output=False, as_head=False, ):
    """
    Calculate the pressure gradient or pressure difference for single-phase fluid flow.

    Parameters
    ----------
    flow_rate: float or array(float)
        flow_rate in m3/s
    *args: additional positional arguments. Unused (for compatibility)
    D: float
        diameter in m.
    density: float
        fluid density in kg/m3
    viscosity: float
        fluid viscosity in Pa.s (default 1)
    inc: float between -1 (full downwards) and 1 (full upwards). 
        NOTE: if L is set, simply setting inc = dz / L will add the height (dz) component 
        Default is 0
    eps: float
        pipe roughness in m (default 0.15 mm)
    compressibility: float
        compressibility of fluid in Pa**-1 (default 0)
    L: float
      length of pipe. Setting L= value 1, which will return the gradient.
      For incompressible fluids setting L can be used to obtain the total loss
      Default is 1.
    K: float
      additional pressure loss factors for elbows, valve, etc.
      Default is 0.
    output_components: bool
      if True, returns the three components of pressure loss (gravity gradient, friction gradient, momentum gradient).
      Otherwise returns the sum.
      Default is False
    full_output: bool
        if True, returns dictionary of intermediate calculations. Default is False.
    as_head: bool
      if True, returns the pressure drop as head (m), otherwise in Pa. 
      Default is False.

    Returns
    -------
    float or tuple or dict
        Total gradient or an array of gradients (dPg, dPf, dPv).
    
        If P0 is set, returns end pressure
        If P1 is set, returns the initial pressure
        If neither is set, returns the pressure difference
    """
    dPg = -inc * L
    A = D**2 / 4 * np.pi
    eD = eps / D
    v = flow_rate / A
    re = reynolds(v, D, density, viscosity)
    f = find_friction_factor(re=re, eD=eD, fanning=False)
    dPf = - (f / D * L + K) * v**2 / (2 * spc.g)
    Eh = compressibility * v**2 / spc.g
    if np.any(Eh >= 1):
        raise ValueError("Supersonic flow encountered.")
    elif np.any(Eh > 0.9):
        warnings.warn("Flow is close to supersonic.")
    dPv = (dPf + dPg) * Eh / (1 - Eh)
    if not as_head:
        dPg *= density * spc.g
        dPf *= density * spc.g
        dPv *= density * spc.g
    
    total_loss = dPg + dPf + dPv
    
    if full_output:
        return dict(total_loss=total_loss, friction_loss=dPf, gravity_loss=dPg, kinetic_loss= dPv, friction_factor=f, reynolds=re, v=v, eD=eD,)
    elif output_components:
        return (dPg, dPf, dPv)
    else:
        return total_loss

single_phase_head_gradient = functools.partial(single_phase_pressure_gradient, as_head=True)