"""Tests for friction factor and single-phase gradient."""

import numpy as np
import pytest

from fluidnet.physics.friction import friction_factor
from fluidnet.physics.single_phase import single_phase_gradient


class TestFrictionFactor:
    def test_positional_d_eps_forbidden(self) -> None:
        """D/eps are keyword-only: the 2018 arg-swap bug cannot recur."""
        with pytest.raises(TypeError):
            friction_factor(1e5, 0.1, 1e-4)  # type: ignore

    def test_laminar(self) -> None:
        assert friction_factor(1000, D=0.1, eps=1e-4) == pytest.approx(16 / 1000)

    def test_darcy_is_4x_fanning(self) -> None:
        f = friction_factor(1e5, D=0.1, eps=1e-4, fanning=True)
        fd = friction_factor(1e5, D=0.1, eps=1e-4, fanning=False)
        assert fd == pytest.approx(4 * f)

    def test_negative_re_raises(self) -> None:
        with pytest.raises(ValueError):
            friction_factor(-1.0, D=0.1, eps=1e-4)

    def test_vectorized_regimes(self) -> None:
        f = friction_factor([500, 3000, 1e5], D=0.1, eps=1e-4)  # type: ignore
        assert np.asarray(f).shape == (3,)
        assert np.all(f > 0)


class TestSinglePhase:
    def test_positional_args_forbidden(self) -> None:
        """All params are keyword-only: no arg-order bug can recur (see CLAUDE.md §8)."""
        with pytest.raises(TypeError):
            single_phase_gradient(10, 0.1, 1000, 1e-3)  # type: ignore

    def test_horizontal_no_gravity(self) -> None:
        res = single_phase_gradient(mass_rate=10, D=0.1, density=1000, viscosity=1e-3)
        assert res.gravity == 0
        assert res.friction < 0
        assert res.total == pytest.approx(res.friction)

    def test_static_column(self) -> None:
        """Zero rate, vertical: pure hydrostatic gradient -rho*g."""
        res = single_phase_gradient(
            mass_rate=0, D=0.1, density=1000, viscosity=1e-3, inclination=1.0
        )
        assert res.friction == 0
        assert res.gravity == pytest.approx(-1000 * 9.80665, rel=1e-4)

    def test_reversed_flow_flips_friction_sign(self) -> None:
        fwd = single_phase_gradient(mass_rate=10, D=0.1, density=1000, viscosity=1e-3)
        rev = single_phase_gradient(mass_rate=-10, D=0.1, density=1000, viscosity=1e-3)
        assert rev.friction == pytest.approx(-fwd.friction)

    def test_supersonic_raises(self) -> None:
        with pytest.raises(ValueError, match="Supersonic"):
            single_phase_gradient(
                mass_rate=500, D=0.05, density=1.2, viscosity=1.8e-5, compressibility=1e-3
            )
