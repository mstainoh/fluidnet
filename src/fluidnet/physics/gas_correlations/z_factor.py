"""Gas compressibility-factor (Z-factor) correlations.

Layer zero: pure SI-in/SI-out (dimensionless Z), no package imports.
"""

from typing import cast

import numpy as np
from scipy.optimize import newton

from fluidnet._types import ArrayLike


def z_hall_yarborough(
    pressure_reduced: ArrayLike,
    temperature_reduced: ArrayLike,
    *,
    tol: float = 1e-8,
    maxiter: int = 50,
) -> ArrayLike:
    """Hall-Yarborough compressibility factor.

    Hall, K.R. & Yarborough, L. (1973). "A new equation of state for
    Z-factor calculations." Oil and Gas Journal, 71(7), 82-92.

    Cross-checked against Ahmed, T., *Reservoir Engineering Handbook*
    (Eqs. 2-36/2-37) and Kareem, Elsharkawy & Alostad (2016), J. Pet.
    Explor. Prod. Technol.

    Vectorized: ``scipy.optimize.newton`` dispatches to its array
    implementation when the initial guess has more than one element, so
    array-valued ``pressure_reduced``/``temperature_reduced`` solve
    element-wise in one call.

    Parameters
    ----------
    pressure_reduced : ArrayLike
        Pseudo-reduced pressure ``Ppr = P / Pc`` (dimensionless).
    temperature_reduced : ArrayLike
        Pseudo-reduced temperature ``Tpr = T / Tc`` (dimensionless). Must
        be greater than 1.0 everywhere.
    tol : float, optional
        Convergence tolerance on ``y``, passed to
        :func:`scipy.optimize.newton`, by default 1e-8.
    maxiter : int, optional
        Maximum iterations, passed to :func:`scipy.optimize.newton`, by
        default 50.

    Returns
    -------
    ArrayLike
        Compressibility factor ``Z`` (dimensionless).

    Raises
    ------
    ValueError
        If any element of ``temperature_reduced`` is ``<= 1.0`` — the
        correlation is not valid there.
    """
    if np.any(np.asarray(temperature_reduced) <= 1.0):
        raise ValueError("Hall-Yarborough is not valid for Tpr <= 1.0")

    t = 1.0 / temperature_reduced
    X1 = 0.06125 * pressure_reduced * t * np.exp(-1.2 * (1.0 - t) ** 2)
    X2 = 14.76 * t - 9.76 * t**2 + 4.58 * t**3
    X3 = 90.7 * t - 242.2 * t**2 + 42.4 * t**3
    X4 = 2.18 + 2.82 * t

    def F(y: ArrayLike, /) -> ArrayLike:
        return cast(
            ArrayLike, -X1 + (y + y**2 + y**3 - y**4) / (1.0 - y) ** 3 - X2 * y**2 + X3 * y**X4
        )

    def dF(y: ArrayLike, /) -> ArrayLike:
        return cast(
            ArrayLike,
            (1 + 4 * y + 4 * y**2 - 4 * y**3 + y**4) / (1.0 - y) ** 4
            - 2 * X2 * y
            + X3 * X4 * y ** (X4 - 1),
        )

    y0 = 0.0125 * pressure_reduced * t * np.exp(-1.2 * (1.0 - t) ** 2)
    y = newton(F, y0, fprime=dF, tol=tol, maxiter=maxiter)
    return cast(ArrayLike, X1 / y)


def z_dranchuk_abou_kassem(
    pressure_reduced: ArrayLike,
    temperature_reduced: ArrayLike,
    *,
    tol: float = 1e-8,
    maxiter: int = 50,
) -> ArrayLike:
    """Dranchuk-Abou-Kassem compressibility factor.

    Dranchuk, P.M. & Abou-Kassem, J.H. (1975). "Calculation of Z Factors
    For Natural Gases Using Equations of State." Journal of Canadian
    Petroleum Technology, 14(3). DOI: 10.2118/75-03-03.

    Eleven-constant, Benedict-Webb-Rubin-type EOS, fitted to 1500 points
    of the Standing-Katz chart (average absolute error ~0.585% per the
    original paper). Valid for ``0.2 <= Ppr < 30``, ``1.0 < Tpr <= 3.0``.
    Cross-checked against Craft, Hawkins & Terry, *Applied Petroleum
    Reservoir Engineering* (2nd ed., 1991).

    Vectorized: see :func:`z_hall_yarborough`. The initial guess is
    broadcast to the ``Ppr``/``Tpr`` shape so array inputs reliably trigger
    ``scipy.optimize.newton``'s array path.

    Parameters
    ----------
    pressure_reduced : ArrayLike
        Pseudo-reduced pressure ``Ppr = P / Pc`` (dimensionless).
    temperature_reduced : ArrayLike
        Pseudo-reduced temperature ``Tpr = T / Tc`` (dimensionless).
    tol : float, optional
        Convergence tolerance on ``Z``, passed to
        :func:`scipy.optimize.newton`, by default 1e-8.
    maxiter : int, optional
        Maximum iterations, passed to :func:`scipy.optimize.newton`, by
        default 50.

    Returns
    -------
    ArrayLike
        Compressibility factor ``Z`` (dimensionless).
    """
    A1, A2, A3, A4, A5 = 0.3265, -1.0700, -0.5339, 0.01569, -0.05165
    A6, A7, A8, A9, A10, A11 = 0.5475, -0.7361, 0.1844, 0.1056, 0.6134, 0.7210

    Tpr = temperature_reduced
    Ppr = pressure_reduced
    C1 = A1 + A2 / Tpr + A3 / Tpr**3 + A4 / Tpr**4 + A5 / Tpr**5
    C2 = A6 + A7 / Tpr + A8 / Tpr**2
    C3 = A9 * (A7 / Tpr + A8 / Tpr**2)

    def f(Z: ArrayLike, /) -> ArrayLike:
        rho_r = 0.27 * Ppr / (Z * Tpr)
        return cast(
            ArrayLike,
            1.0
            + C1 * rho_r
            + C2 * rho_r**2
            - C3 * rho_r**5
            + A10 * (1 + A11 * rho_r**2) * (rho_r**2 / Tpr**3) * np.exp(-A11 * rho_r**2)
            - Z,
        )

    Z0 = np.ones_like(Ppr * Tpr, dtype=float)
    Z = newton(f, Z0, tol=tol, maxiter=maxiter)
    return cast(ArrayLike, Z)
