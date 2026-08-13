"""Tests for IdealGas and RealGas (CompressibleFluid) — methane at STP."""

import numpy as np
import pytest
import scipy.constants as spc

from fluidnet.state.fluids import IdealGas, RealGas

METHANE_MOLAR_WEIGHT = 16.043e-3  # kg/mol
METHANE_VISCOSITY = 10.84e-6  # Pa*s, assumed constant
STANDARD_TEMPERATURE = 273.15  # K, 0 degC
STANDARD_PRESSURE = np.array([spc.atm])  # Pa, 1 atm (bound() indexes across[0])
METHANE_DENSITY_STP = 0.717  # kg/m3, literature value at 0 degC / 1 atm


class TestIdealGasMethane:
    def _fluid(self) -> IdealGas:
        return IdealGas(
            molar_weight=METHANE_MOLAR_WEIGHT,
            viscosity=lambda pressure, temperature: METHANE_VISCOSITY,
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
            molar_weight=METHANE_MOLAR_WEIGHT,
            viscosity=lambda pressure, temperature: METHANE_VISCOSITY,
        )

    def _real(self) -> RealGas:
        return RealGas(
            molar_weight=METHANE_MOLAR_WEIGHT,
            z_fn=lambda pressure, temperature: 1.0,
            dz_fn=lambda pressure, temperature: 0.0,
            viscosity_fn=lambda pressure, temperature: METHANE_VISCOSITY,
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
