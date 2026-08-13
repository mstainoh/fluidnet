from __future__ import annotations

from collections.abc import Callable

import scipy.constants as spc

from fluidnet.physics.types import ArrayLike

from .single_phase_fluids import CompressibleFluid


class IdealGas(CompressibleFluid):
    """Ideal-gas EOS: rho = P/(R*T) (#4)."""

    def __init__(
        self, *, molar_weight: float, viscosity: Callable[[ArrayLike, float], float]
    ) -> None:
        self.molar_weight = molar_weight
        self._viscosity_fn = viscosity

    def density(self, *, pressure: ArrayLike, temperature: float | None = None) -> ArrayLike:
        if temperature is None:
            raise ValueError("Temperature must be provided for ideal gas density calculation.")
        R_specific = spc.R / self.molar_weight  # Specific gas constant
        return pressure / (R_specific * temperature)

    def compressibility(
        self, *, pressure: ArrayLike, temperature: float | None = None
    ) -> ArrayLike:
        if temperature is None:
            raise ValueError(
                "Temperature must be provided for ideal gas compressibility calculation."
            )
        # beta = (1/rho)(d rho/dP)_T; for an ideal gas rho is linear in P at
        # fixed T, so this collapses to 1/P regardless of R_specific/T.
        return 1 / pressure

    def viscosity(self, *, pressure: ArrayLike, temperature: float | None = None) -> ArrayLike:
        if temperature is None:
            raise ValueError("Temperature must be provided for ideal gas viscosity calculation.")
        return self._viscosity_fn(pressure, temperature)


class RealGas(CompressibleFluid):
    """Real-gas EOS: rho = f(P,T) (#4)."""

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
        return self.Pr is not None and self.Tr is not None

    def z(self, *, pressure: ArrayLike, temperature: float) -> ArrayLike:
        P, T = pressure, temperature
        if self.Pr is not None and self.Tr is not None:
            P = P / self.Pr
            T = T / self.Tr
        return self.z_fn(P, T)

    def density(self, *, pressure: ArrayLike, temperature: float | None = None) -> ArrayLike:
        if temperature is None:
            raise ValueError("Temperature must be provided for real gas density calculation.")
        z = self.z(pressure=pressure, temperature=temperature)
        R_specific = spc.R / self.molar_weight
        return pressure / (z * R_specific * temperature)

    def compressibility(
        self, *, pressure: ArrayLike, temperature: float | None = None
    ) -> ArrayLike:
        if temperature is None:
            raise ValueError(
                "Temperature must be provided for real gas compressibility calculation."
            )
        # beta = 1/P - (1/Z)(dZ/dP)_T, from rho = P/(Z R_specific T).
        z = self.z(pressure=pressure, temperature=temperature)
        dz_dP = self.dz_fn(pressure, temperature)
        return 1 / pressure - 1 / z * dz_dP

    def viscosity(self, *, pressure: ArrayLike, temperature: float | None = None) -> ArrayLike:
        if temperature is None:
            raise ValueError("Temperature must be provided for real gas viscosity calculation.")
        return self.viscosity_fn(pressure, temperature)
