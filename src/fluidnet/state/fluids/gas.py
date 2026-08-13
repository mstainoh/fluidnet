from __future__ import annotations

from collections.abc import Callable

import scipy.constants as spc

from fluidnet.physics.types import ArrayLike

from .single_phase_fluids import CompressibleFluid


class IdealGas(CompressibleFluid):
    """Ideal-gas EOS: ``rho = P / (R_specific * T)`` (#4).

    Parameters
    ----------
    molar_weight : float
        Molar weight of the gas [kg/mol].
    viscosity : Callable[[ArrayLike, float], ArrayLike]
        Viscosity model, ``viscosity(pressure, temperature) -> mu`` [Pa.s].
    """

    def __init__(
        self, *, molar_weight: float, viscosity: Callable[[ArrayLike, float], float]
    ) -> None:
        self.molar_weight = molar_weight
        self._viscosity_fn = viscosity

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
        R_specific = spc.R / self.molar_weight  # Specific gas constant
        return pressure / (R_specific * temperature)

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

    def viscosity(self, *, pressure: ArrayLike, temperature: float | None = None) -> ArrayLike:
        """Ideal-gas viscosity, delegated to the constructor's model.

        Parameters
        ----------
        pressure : ArrayLike
            Pressure [Pa].
        temperature : float, optional
            Temperature [K]. Required — ``None`` raises ``ValueError``.

        Returns
        -------
        ArrayLike
            Dynamic viscosity [Pa.s].
        """
        if temperature is None:
            raise ValueError("Temperature must be provided for ideal gas viscosity calculation.")
        return self._viscosity_fn(pressure, temperature)


class RealGas(CompressibleFluid):
    """Real-gas EOS: ``rho = P / (Z * R_specific * T)`` (#4).

    Parameters
    ----------
    molar_weight : float
        Molar weight of the gas [kg/mol].
    z_fn : Callable[[ArrayLike, float], ArrayLike]
        Compressibility-factor model, ``z_fn(pressure, temperature) -> Z``
        (dimensionless). Evaluated at reduced ``(P/Pr, T/Tr)`` when ``Pr``
        and ``Tr`` are given, otherwise at ``(P, T)`` directly.
    dz_fn : Callable[[ArrayLike, float], ArrayLike]
        Derivative model, ``dz_fn(pressure, temperature) -> dZ/dP`` [1/Pa].
        Used by :meth:`compressibility`.
    viscosity_fn : Callable[[ArrayLike, float], ArrayLike]
        Viscosity model, ``viscosity_fn(pressure, temperature) -> mu``
        [Pa.s].
    Pr : float, optional
        Pseudo-critical pressure [Pa], for reduced-property correlations.
        Must be given together with ``Tr``, or not at all.
    Tr : float, optional
        Pseudo-critical temperature [K], for reduced-property correlations.
        Must be given together with ``Pr``, or not at all.
    """

    def __init__(
        self,
        *,
        molar_weight: float,
        z_fn: Callable[[ArrayLike, float], ArrayLike],
        dz_fn: Callable[[ArrayLike, float], ArrayLike],
        viscosity_fn: Callable[[ArrayLike, float], ArrayLike],
        Pr: float | None = None,
        Tr: float | None = None,
    ) -> None:
        self.molar_weight = molar_weight
        self.Pr, self.Tr = Pr, Tr
        assert (Pr is None) == (Tr is None), "Both Pr and Tr must be provided together, or ignored."
        self.z_fn = z_fn
        self.dz_fn = dz_fn
        self.viscosity_fn = viscosity_fn

    @property
    def uses_reduced_properties(self) -> bool:
        """bool: Whether ``z_fn``/``dz_fn`` are evaluated at reduced ``(P, T)``
        (both ``Pr`` and ``Tr`` given) rather than at absolute ``(P, T)``.
        """
        return self.Pr is not None and self.Tr is not None

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
        P, T = pressure, temperature
        if self.Pr is not None and self.Tr is not None:
            P = P / self.Pr
            T = T / self.Tr
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
        R_specific = spc.R / self.molar_weight
        return pressure / (z * R_specific * temperature)

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
        dz_dP = self.dz_fn(pressure, temperature)
        return 1 / pressure - 1 / z * dz_dP

    def viscosity(self, *, pressure: ArrayLike, temperature: float | None = None) -> ArrayLike:
        """Real-gas viscosity, delegated to the constructor's model.

        Parameters
        ----------
        pressure : ArrayLike
            Pressure [Pa].
        temperature : float, optional
            Temperature [K]. Required — ``None`` raises ``ValueError``.

        Returns
        -------
        ArrayLike
            Dynamic viscosity [Pa.s].
        """
        if temperature is None:
            raise ValueError("Temperature must be provided for real gas viscosity calculation.")
        return self.viscosity_fn(pressure, temperature)
