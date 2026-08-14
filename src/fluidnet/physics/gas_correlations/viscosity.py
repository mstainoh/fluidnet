"""Gas viscosity correlations. Layer zero: pure SI->SI, no package imports."""

import numpy as np

from fluidnet._types import ArrayLike


def sutherland_viscosity(
    pressure: ArrayLike,
    temperature: ArrayLike,
    *,
    mu_ref: float,
    T_ref: float,
    S: float,
    **kwargs: ArrayLike,
) -> ArrayLike:
    """Sutherland (1893) viscosity for a pure gas.

    Parameters
    ----------
    pressure : ArrayLike
        Pressure [Pa]. Unused — accepted for protocol parity.
    temperature : ArrayLike
        Temperature [K].
    mu_ref : float
        Reference dynamic viscosity at ``T_ref`` [Pa.s].
    T_ref : float
        Reference temperature [K].
    S : float
        Sutherland constant [K].
    **kwargs : ArrayLike
        Injectables not used by this model (``density``, ``molecular_weight``).

    Returns
    -------
    ArrayLike
        Dynamic viscosity [Pa.s].
    """
    return mu_ref * (T_ref + S) / (temperature + S) * (temperature / T_ref) ** 1.5


def _lee_gonzalez_eakin_detailed(
    pressure: ArrayLike,
    temperature: ArrayLike,
    *,
    density: ArrayLike,
    molecular_weight: float,
    **kwargs: ArrayLike,
) -> dict[str, ArrayLike]:
    """Full Lee, Gonzalez & Eakin (1966) calculation, returning viscosity
    plus intermediates.

    Internal: consumed by golden tests (``CLAUDE.md`` #25 pattern — same
    shape as ``_beggs_brill_detailed`` — applied here at the correlation
    level rather than the gradient level).
    :func:`lee_gonzalez_eakin_viscosity` is a thin wrapper that extracts
    ``viscosity`` from this dict.

    Parameters
    ----------
    pressure : ArrayLike
        Pressure [Pa]. Unused — the pressure dependence enters through
        ``density``.
    temperature : ArrayLike
        Temperature [K].
    density : ArrayLike
        Gas density [kg/m3]. Must be the *real* density (Z included) when
        the caller is a :class:`RealGas`.
    molecular_weight : float
        Apparent molecular weight [kg/mol].
    **kwargs : ArrayLike
        Injectables not used by this model.

    Returns
    -------
    dict[str, ArrayLike]
        ``viscosity`` [Pa.s] plus intermediates ``K``, ``X``, ``Y`` (Ahmed,
        Eqs. 2-63/2-64/2-65 — dimensionless-ish field-unit coefficients).
    """
    M = molecular_weight * 1e3
    T_R = temperature * 1.8
    rho = density * 1e-3

    K = (9.4 + 0.02 * M) * T_R**1.5 / (209.0 + 19.0 * M + T_R)
    X = 3.5 + 986.0 / T_R + 0.01 * M
    Y = 2.4 - 0.2 * X

    mu_cP = 1e-4 * K * np.exp(X * rho**Y)

    return {"viscosity": mu_cP * 1e-3, "K": K, "X": X, "Y": Y}


def lee_gonzalez_eakin_viscosity(
    pressure: ArrayLike,
    temperature: ArrayLike,
    *,
    density: ArrayLike,
    molecular_weight: float,
    **kwargs: ArrayLike,
) -> ArrayLike:
    """Lee, Gonzalez & Eakin (1966) viscosity for sweet natural gas.

    Parameters
    ----------
    pressure : ArrayLike
        Pressure [Pa]. Unused — the pressure dependence enters through
        ``density``.
    temperature : ArrayLike
        Temperature [K].
    density : ArrayLike
        Gas density [kg/m3]. Must be the *real* density (Z included) when
        the caller is a :class:`RealGas`.
    molecular_weight : float
        Apparent molecular weight [kg/mol].
    **kwargs : ArrayLike
        Injectables not used by this model.

    Returns
    -------
    ArrayLike
        Dynamic viscosity [Pa.s]. See :func:`_lee_gonzalez_eakin_detailed`
        for the intermediates (``K``, ``X``, ``Y``).

    Notes
    -----
    Derived for sweet gas only. Applying it to sour gas (CO2/N2/H2S) is a
    known misuse — use a CKB-family correlation there.

    The correlation is defined in field units with an explicit ``1e-4``
    prefactor (Ahmed, Eq. 2-63): ``mu[cP] = 1e-4 * K * exp(X * rho^Y)``,
    with ``T`` [R], ``rho`` [g/cm3], ``M`` [g/mol]. Conversions are
    internal; the SI result is [Pa.s].
    """
    return _lee_gonzalez_eakin_detailed(
        pressure, temperature, density=density, molecular_weight=molecular_weight, **kwargs
    )["viscosity"]
