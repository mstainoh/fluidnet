"""Friction factor (Chen approximation), laminar/transition/turbulent.

``D`` and ``eps`` are keyword-only on purpose: the 2018 prototype defined
``(re, D, eps)`` but called ``(re, eps, D)`` at both call sites — an argument
swap that goes numerically unnoticed when roughness << diameter. Keyword-only
arguments make that class of bug impossible.
"""
import numpy as np

from fluidnet._types import ArrayLike


def _chen_approx(re: ArrayLike, D: ArrayLike, eps: ArrayLike) -> ArrayLike:
    """Chen (1979) explicit approximation for turbulent Fanning friction.

    Parameters
    ----------
    re : ArrayLike
        Reynolds number.
    D : ArrayLike
        Pipe diameter [m].
    eps : ArrayLike
        Absolute roughness [m], same units as ``D``.

    Returns
    -------
    ArrayLike
        Fanning friction factor (turbulent correlation).
    """
    re = np.clip(re, 2e3, None)
    return (
        -4
        * np.log10(
            0.2698 * (eps / D)
            - 5.0452 / re * np.log10(0.3539 * (eps / D) ** 1.1098 + 5.8506 / re**0.8981)
        )
    ) ** -2


def friction_factor(
        re: ArrayLike, *,
        D: ArrayLike,
        eps: ArrayLike,
        fanning: bool = True
        ) -> ArrayLike:
    """Friction factor for pipe flow. Vectorized over ``re``.

    Laminar (``re <= 2000``): ``16 / re``. Turbulent (``re >= 4000``): Chen
    approximation. Transition: linear blend of both.

    Parameters
    ----------
    re : ArrayLike
        Reynolds number (must be >= 0).
    D : ArrayLike
        Pipe diameter [m]. Keyword-only.
    eps : ArrayLike
        Absolute roughness [m], same units as ``D``. Keyword-only.
    fanning : bool, optional
        Return Fanning factor (default). ``False`` returns Darcy-Weisbach
        (``f_DW = 4 * f_Fanning``).

    Returns
    -------
    ArrayLike
        Friction factor (Fanning by default, Darcy-Weisbach if
        ``fanning=False``).
    """
    re = np.asarray(re, dtype=float)
    if np.any(re < 0):
        raise ValueError("Reynolds number cannot be negative")

    f = np.zeros_like(re)
    laminar = (re > 0) & (re <= 2000)
    transition = (re > 2000) & (re < 4000)
    turbulent = re >= 4000

    with np.errstate(divide="ignore"):
        f_lam = np.where(re > 0, 16 / np.where(re > 0, re, 1.0), 0.0)
    f_turb = _chen_approx(re, D, eps)

    np.putmask(f, laminar, f_lam)
    np.putmask(f, turbulent, f_turb)
    # linear blend across 2000 < re < 4000
    blend = (f_turb * (re - 2e3) + f_lam * (4e3 - re)) / 2e3
    np.putmask(f, transition, blend)

    f = f if fanning else f * 4
    return float(f) if f.ndim == 0 else f
