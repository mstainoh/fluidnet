"""
inverse_solvers.py

This module provides flexible numeric inversion tools for monotonic or mostly-monotonic functions.
It includes two root-finding wrappers:

- `brent_inverse`: Robust and reliable inversion using Brent’s method, with dynamic bracketing.
- `newton_inverse`: Faster inversion using Newton’s method, with optional fallback to Brent if convergence fails.

Both functions support scalar and vector inputs and pass extra arguments to the target function via `**kwargs`.

Example usage:
    def square(x): return x**2
    x = brent_inverse(9, square, low=0.1, high=1)  # Returns ~3.0
    x = newton_inverse(9, square, low=0.1, high=5) # Returns ~3.0

Exports:
    - brent_inverse
    - newton_inverse
"""

from scipy.optimize import newton, brentq
import numpy as np

__all__ = ['brent_inverse', 'newton_inverse']

def brent_inverse(
    y, func,
    low, high,
    lowscale=10, highscale=100,
    lowlimit=1e-6, highlimit=1e8, 
    **kwargs
):
    """
    Inverts a function using Brent’s method with automatic bracketing expansion.

    This method is robust and does not require the function to be differentiable.
    It works for scalar or vector `y` inputs.

    Parameters:
        y : float or array-like
            Target value(s) for which func(x) ≈ y.
        func : callable
            Monotonic or mostly monotonic function to invert.
        low : float
            Initial lower bound for root search.
        high : float
            Initial upper bound for root search.
        lowscale : float, default=10
            Factor to divide the lower bound if bracketing fails.
        highscale : float, default=100
            Factor to multiply the upper bound if bracketing fails.
        lowlimit : float, default=1e-6
            Minimum allowed lower bound for expansion.
        highlimit : float, default=1e8
            Maximum allowed upper bound for expansion.
        **kwargs :
            Additional arguments passed to `func`.

    Returns:
        float or np.ndarray
            Value(s) x such that func(x) ≈ y.

    Raises:
        RuntimeError if a valid bracketing interval cannot be found.
    """
    if np.ndim(y):  # array-like
        return np.array([brent_inverse(
            yy, func, low=low, high=high, 
            lowscale=lowscale, highscale=highscale,
            lowlimit=lowlimit, highlimit=highlimit, 
            **kwargs)
          for yy in y])

    def f(x):
        return func(x, **kwargs) - y

    f_low = f(low)
    f_high = f(high)

    while np.sign(f_low) == np.sign(f_high):
        high *= highscale
        low /= lowscale
        f_high = f(high)
        f_low = f(low)
        if high > highlimit and low < lowlimit:
            raise RuntimeError("Failed to bracket root - max/min bracket limit reached.")

    return brentq(f, low, high)


def newton_inverse(
    y, func, low, high,
    fallback=True,
    newton_parameters=dict(),
    **kwargs
):
    """
    Inverts a (preferably monotonic and smooth) function using Newton’s method,
    optionally falling back to Brent’s method if Newton fails to converge.

    Works for scalar or vector `y` inputs.

    Parameters:
        y : float or array-like
            Target value(s) for which func(x) ≈ y.
        func : callable
            Function to invert.
        low : float
            Lower bound used for fallback bracketing or initial guess.
        high : float
            Upper bound used for fallback bracketing or initial guess.
        fallback : bool, default=True
            Whether to fall back to Brent's method if Newton fails.
        **kwargs :
            Additional arguments passed to `func`.

    Returns:
        float or np.ndarray
            Value(s) x such that func(x) ≈ y.

    Raises:
        RuntimeError if both Newton and fallback Brent fail.
    """
    if np.ndim(y):
        return np.array([
            newton_inverse(yy, func, low, high,
                           newton_parameters=newton_parameters,
                           fallback=fallback, **kwargs)
            for yy in y
        ])

    def f(x):
        return func(x, **kwargs) - y

    try:
        guess = (low + high) / 2
        return newton(f, x0=guess, **newton_parameters)
    except (RuntimeError, OverflowError):
        if not fallback:
            raise
        return brentq(f, low, high)
