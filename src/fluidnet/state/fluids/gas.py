"""Provides convenience class implementations for gas. Two classes:

``IdealGas``: instantiated with a molecular weight and a viscosity function
(see ``physics.gas_correlations``, or a user-supplied one).

``RealGas``: adds a ``z_fn`` (compressibility factor) and its derivative
``dz_fn``, supplied by the caller. If critical values ``Pc``/``Tc`` are
given, both are evaluated at reduced pressure/temperature instead of
absolute values.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import scipy.constants as spc

from fluidnet._types import ArrayLike

from .single_phase import CompressibleFluidBase


class IdealGas(CompressibleFluidBase):
    """Ideal-gas EOS: ``rho = P / (R_specific * T)`` (#4).

    Parameters
    ----------
    molecular_weight : float
        Molecular weight of the gas [kg/mol].
    viscosity_fn : Callable[..., ArrayLike]
        Viscosity correlation, ``viscosity_fn(pressure, temperature,
        **injectables) -> mu`` [Pa.s] (``CLAUDE.md`` #21). ``IdealGas``
        injects ``density`` [kg/m3] (the ideal-gas density) and
        ``molecular_weight`` [kg/mol] — no reduced properties, since an
        ideal EOS has no ``Pc``/``Tc``. Injected parameters must be
        declared keyword-only with no default in ``viscosity_fn``'s own
        signature — a naming typo then fails loudly with ``TypeError``
        instead of silently falling back to a default. A catch-all
        ``**kwargs`` is fine, to ignore injectables the correlation
        doesn't use — a ``T``-only correlation (e.g. Sutherland, valid in
        the dilute-gas limit) simply doesn't declare ``density``/
        ``molecular_weight`` and catches them there. The ideal EOS does
        not force a ``T``-only viscosity model; that's a property of the
        correlation, not of this class. Anything private to the
        correlation (``mu_ref``, ``S``, …) is fixed via
        :func:`functools.partial` before passing ``viscosity_fn`` in.
    """

    def __init__(self, *, molecular_weight: float, viscosity_fn: Callable[..., ArrayLike]) -> None:
        self.molecular_weight = molecular_weight
        self.viscosity_fn = viscosity_fn

    def density(self, *, pressure: ArrayLike, temperature: float | None = None) -> ArrayLike:
        """Ideal-gas density.

        Parameters
        ----------
        pressure : ArrayLike
            Pressure [Pa].
        temperature : float, optional
            Temperature [K]. Required — ``None`` raises ``ValueError``.

        Returns
        -------
        ArrayLike
            Density [kg/m3].
        """
        if temperature is None:
            raise ValueError("Temperature must be provided for ideal gas density calculation.")
        R_specific = spc.R / self.molecular_weight  # Specific gas constant
        return cast(ArrayLike, pressure / (R_specific * temperature))

    def compressibility(
        self, *, pressure: ArrayLike, temperature: float | None = None
    ) -> ArrayLike:
        """Ideal-gas isothermal compressibility.

        Parameters
        ----------
        pressure : ArrayLike
            Pressure [Pa].
        temperature : float, optional
            Temperature [K]. Required — ``None`` raises ``ValueError`` — even
            though the result is ``T``-independent, for interface parity
            with :meth:`density`/:meth:`viscosity`.

        Returns
        -------
        ArrayLike
            Compressibility ``beta = (1/rho)(d rho/dP)_T`` [1/Pa].
        """
        if temperature is None:
            raise ValueError(
                "Temperature must be provided for ideal gas compressibility calculation."
            )
        # beta = (1/rho)(d rho/dP)_T; for an ideal gas rho is linear in P at
        # fixed T, so this collapses to 1/P regardless of R_specific/T.
        return 1 / pressure


class RealGas(CompressibleFluidBase):
    """Real-gas EOS: ``rho = P / (Z * R_specific * T)`` (#4).

    Parameters
    ----------
    molecular_weight : float
        Molecular weight of the gas [kg/mol].
    z_fn : Callable[[ArrayLike, float], ArrayLike]
        Compressibility-factor model, ``z_fn(pressure, temperature) -> Z``
        (dimensionless). Evaluated at reduced ``(P/Pc, T/Tc)`` when ``Pc``
        and ``Tc`` are given, otherwise at ``(P, T)`` directly.
    dz_fn : Callable[[ArrayLike, float], ArrayLike]
        Derivative model, ``dz_fn(pressure, temperature) -> dZ/dP`` [1/Pa].
        Used by :meth:`compressibility`. Receives the **same** ``(P, T)``
        as ``z_fn`` — reduced when ``Pc``/``Tc`` are given, absolute
        otherwise — but must still *return* ``dZ/dP`` with respect to
        *absolute* pressure [1/Pa], matching the formula in
        :meth:`compressibility`. If the underlying correlation is written
        analytically in reduced coordinates (``dZ/d(P/Pc)``), the
        ``1/Pc`` chain-rule factor is the correlation author's
        responsibility to fold in — ``RealGas`` does not apply it.
    viscosity_fn : Callable[..., ArrayLike]
        Viscosity correlation, ``viscosity_fn(pressure, temperature,
        **injectables) -> mu`` [Pa.s] (``CLAUDE.md`` #21). ``RealGas``
        injects what it already knows or has computed: ``density``
        [kg/m3], ``molecular_weight`` [kg/mol], and, when
        :attr:`uses_reduced_properties` is true, ``pressure_reduced`` /
        ``temperature_reduced`` (dimensionless). These injected parameters
        must be declared keyword-only with no default in ``viscosity_fn``'s
        own signature — a typo in the name then fails loudly with
        ``TypeError`` instead of silently falling back to a default. A
        catch-all ``**kwargs`` is fine, to ignore injectables the
        correlation doesn't use. Anything private to the correlation
        itself (e.g. ``mu_ref``, ``S`` in a Sutherland model) is not
        injected — fix it with :func:`functools.partial` before passing
        ``viscosity_fn`` in.
    Pc : float, optional
        Pseudo-critical pressure [Pa], for reduced-property correlations.
        Must be given together with ``Tc``, or not at all.
    Tc : float, optional
        Pseudo-critical temperature [K], for reduced-property correlations.
        Must be given together with ``Pc``, or not at all.
    """

    def __init__(
        self,
        *,
        molecular_weight: float,
        z_fn: Callable[[ArrayLike, float], ArrayLike],
        dz_fn: Callable[[ArrayLike, float], ArrayLike],
        viscosity_fn: Callable[..., ArrayLike],
        Pc: float | None = None,
        Tc: float | None = None,
    ) -> None:
        self.molecular_weight = molecular_weight
        self.Pc, self.Tc = Pc, Tc
        assert (Pc is None) == (Tc is None), "Both Pc and Tc must be provided together, or ignored."
        self.z_fn = z_fn
        self.dz_fn = dz_fn
        self.viscosity_fn = viscosity_fn

    @property
    def uses_reduced_properties(self) -> bool:
        """bool: Whether ``z_fn``/``dz_fn`` are evaluated at reduced ``(P, T)``
        (both ``Pc`` and ``Tc`` given) rather than at absolute ``(P, T)``.
        """
        return self.Pc is not None and self.Tc is not None

    def _reduced_pt(self, *, pressure: ArrayLike, temperature: float) -> tuple[ArrayLike, float]:
        """Map ``(pressure, temperature)`` to reduced coordinates when
        ``Pc``/``Tc`` are given, otherwise pass them through unchanged.
        Shared by :meth:`z` and :meth:`compressibility` so ``z_fn`` and
        ``dz_fn`` always see the same coordinate convention.
        """
        if self.Pc is not None and self.Tc is not None:
            return pressure / self.Pc, temperature / self.Tc
        return pressure, temperature

    def _reduced_injectables(
        self, *, pressure: ArrayLike, temperature: float
    ) -> dict[str, ArrayLike]:
        """``pressure_reduced``/``temperature_reduced`` for ``viscosity_fn``
        (#21), when :attr:`uses_reduced_properties` is true.
        """
        if self.Pc is not None and self.Tc is not None:
            return {
                "pressure_reduced": pressure / self.Pc,
                "temperature_reduced": temperature / self.Tc,
            }
        return {}

    def z(self, *, pressure: ArrayLike, temperature: float) -> ArrayLike:
        """Compressibility factor at the given conditions.

        Parameters
        ----------
        pressure : ArrayLike
            Pressure [Pa].
        temperature : float
            Temperature [K].

        Returns
        -------
        ArrayLike
            Compressibility factor ``Z`` (dimensionless), from ``z_fn``.
        """
        P, T = self._reduced_pt(pressure=pressure, temperature=temperature)
        return self.z_fn(P, T)

    def density(self, *, pressure: ArrayLike, temperature: float | None = None) -> ArrayLike:
        """Real-gas density.

        Parameters
        ----------
        pressure : ArrayLike
            Pressure [Pa].
        temperature : float, optional
            Temperature [K]. Required — ``None`` raises ``ValueError``.

        Returns
        -------
        ArrayLike
            Density [kg/m3].
        """
        if temperature is None:
            raise ValueError("Temperature must be provided for real gas density calculation.")
        z = self.z(pressure=pressure, temperature=temperature)
        R_specific = spc.R / self.molecular_weight
        return cast(ArrayLike, pressure / (z * R_specific * temperature))

    def compressibility(
        self, *, pressure: ArrayLike, temperature: float | None = None
    ) -> ArrayLike:
        """Real-gas isothermal compressibility.

        Parameters
        ----------
        pressure : ArrayLike
            Pressure [Pa].
        temperature : float, optional
            Temperature [K]. Required — ``None`` raises ``ValueError``.

        Returns
        -------
        ArrayLike
            Compressibility ``beta = 1/P - (1/Z)(dZ/dP)_T`` [1/Pa], derived
            from ``rho = P / (Z * R_specific * T)``.
        """
        if temperature is None:
            raise ValueError(
                "Temperature must be provided for real gas compressibility calculation."
            )
        # beta = 1/P - (1/Z)(dZ/dP)_T, from rho = P/(Z R_specific T).
        z = self.z(pressure=pressure, temperature=temperature)
        P, T = self._reduced_pt(pressure=pressure, temperature=temperature)
        dz_dP = self.dz_fn(P, T)
        return 1 / pressure - 1 / z * dz_dP
