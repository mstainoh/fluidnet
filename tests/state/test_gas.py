"""Tests for IdealGas and RealGas (CompressibleFluidBase) — methane at STP."""

from collections.abc import Callable

import numpy as np
import pytest
import scipy.constants as spc

from fluidnet._types import ArrayLike
from fluidnet.physics.gas_correlations import z_dranchuk_abou_kassem, z_hall_yarborough
from fluidnet.state.fluids import IdealGas, RealGas

METHANE_MOLAR_WEIGHT = 16.043e-3  # kg/mol
METHANE_VISCOSITY = 10.84e-6  # Pa*s, assumed constant
STANDARD_TEMPERATURE = 273.15  # K, 0 degC
STANDARD_PRESSURE = np.array([spc.atm])  # Pa, 1 atm (bound() indexes across[0])
METHANE_DENSITY_STP = 0.717  # kg/m3, literature value at 0 degC / 1 atm


class TestIdealGasMethane:
    def _fluid(self) -> IdealGas:
        return IdealGas(
            molecular_weight=METHANE_MOLAR_WEIGHT,
            viscosity_fn=lambda pressure, temperature, **injectables: METHANE_VISCOSITY,
        )

    def test_density_matches_methane_at_standard_conditions(self) -> None:
        state = self._fluid().bind(temperature=STANDARD_TEMPERATURE)(
            x=0.0, across=STANDARD_PRESSURE
        )
        assert state.density == pytest.approx(METHANE_DENSITY_STP, rel=1e-2)

    def test_viscosity_is_constant(self) -> None:
        state = self._fluid().bind(temperature=STANDARD_TEMPERATURE)(
            x=0.0, across=STANDARD_PRESSURE
        )
        assert state.viscosity == METHANE_VISCOSITY

    def test_compressibility_equals_inverse_pressure(self) -> None:
        state = self._fluid().bind(temperature=STANDARD_TEMPERATURE)(
            x=0.0, across=STANDARD_PRESSURE
        )
        assert state.compressibility == pytest.approx(1 / STANDARD_PRESSURE[0])


class TestRealGasMatchesIdealGasWhenZIsOne:
    """z=1 (constant, dZ/dP=0) is the degenerate case of a real gas: it must
    collapse exactly onto the ideal-gas EOS at the same (P, T, M)."""

    def _ideal(self) -> IdealGas:
        return IdealGas(
            molecular_weight=METHANE_MOLAR_WEIGHT,
            viscosity_fn=lambda pressure, temperature, **injectables: METHANE_VISCOSITY,
        )

    def _real(self) -> RealGas:
        return RealGas(
            molecular_weight=METHANE_MOLAR_WEIGHT,
            z_fn=lambda pressure, temperature: 1.0,
            dz_fn=lambda pressure, temperature: 0.0,
            viscosity_fn=lambda pressure, temperature, **injectables: METHANE_VISCOSITY,
        )

    def test_density_matches_methane_at_standard_conditions(self) -> None:
        state = self._real().bind(temperature=STANDARD_TEMPERATURE)(x=0.0, across=STANDARD_PRESSURE)
        assert state.density == pytest.approx(METHANE_DENSITY_STP, rel=1e-2)

    def test_density_matches_ideal_gas(self) -> None:
        ideal_state = self._ideal().bind(temperature=STANDARD_TEMPERATURE)(
            x=0.0, across=STANDARD_PRESSURE
        )
        real_state = self._real().bind(temperature=STANDARD_TEMPERATURE)(
            x=0.0, across=STANDARD_PRESSURE
        )
        assert real_state.density == pytest.approx(ideal_state.density)

    def test_compressibility_matches_ideal_gas(self) -> None:
        ideal_state = self._ideal().bind(temperature=STANDARD_TEMPERATURE)(
            x=0.0, across=STANDARD_PRESSURE
        )
        real_state = self._real().bind(temperature=STANDARD_TEMPERATURE)(
            x=0.0, across=STANDARD_PRESSURE
        )
        assert real_state.compressibility == pytest.approx(ideal_state.compressibility)

    def test_viscosity_matches_ideal_gas(self) -> None:
        state = self._real().bind(temperature=STANDARD_TEMPERATURE)(x=0.0, across=STANDARD_PRESSURE)
        assert state.viscosity == METHANE_VISCOSITY


class TestRealGasZWhenZIsNotOne:
    """``RealGas.z()`` must be a transparent pass-through to ``z_fn`` at the
    reduced ``(Ppr, Tpr)`` it derives from ``(P, T, Pc, Tc)`` — not a
    rescaled or reinterpreted value. Checked against both correlations from
    ``physics/gas_correlations/z_factor.py`` separately, at a condition well
    inside their valid range and away from the near-critical region
    (``Tpr`` close to 1.0) where ``tests/physics/test_z_factor_vs_book.py``
    shows both are numerically rougher.
    """

    PC = 4.599e6  # Pa, methane pseudo-critical pressure
    TC = 190.6  # K, methane pseudo-critical temperature
    PRESSURE = 6.0e6  # Pa
    TEMPERATURE = 300.0  # K

    def _real(self, z_fn: Callable[[ArrayLike, ArrayLike], ArrayLike]) -> RealGas:
        return RealGas(
            molecular_weight=METHANE_MOLAR_WEIGHT,
            z_fn=z_fn,
            dz_fn=lambda pressure, temperature: 0.0,  # unused by .z()
            viscosity_fn=lambda pressure, temperature, **injectables: METHANE_VISCOSITY,
            Pc=self.PC,
            Tc=self.TC,
        )

    def test_z_matches_hall_yarborough(self) -> None:
        real = self._real(z_hall_yarborough)
        Ppr, Tpr = self.PRESSURE / self.PC, self.TEMPERATURE / self.TC
        expected = z_hall_yarborough(Ppr, Tpr)
        assert real.z(pressure=self.PRESSURE, temperature=self.TEMPERATURE) == expected

    def test_z_matches_dranchuk_abou_kassem(self) -> None:
        real = self._real(z_dranchuk_abou_kassem)
        Ppr, Tpr = self.PRESSURE / self.PC, self.TEMPERATURE / self.TC
        expected = z_dranchuk_abou_kassem(Ppr, Tpr)
        assert real.z(pressure=self.PRESSURE, temperature=self.TEMPERATURE) == expected
