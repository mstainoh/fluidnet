"""Tests for friction factor and single-phase gradient."""

import numpy as np
import pytest

from fluidnet.physics.friction import friction_factor
from fluidnet.physics.single_phase import single_phase_gradient


class TestFrictionFactor:
    def test_positional_d_eps_forbidden(self):
        """D/eps are keyword-only: the 2018 arg-swap bug cannot recur."""
        with pytest.raises(TypeError):
            friction_factor(1e5, 0.1, 1e-4)  # noqa: B026

    def test_laminar(self):
        assert friction_factor(1000, D=0.1, eps=1e-4) == pytest.approx(16 / 1000)

    def test_darcy_is_4x_fanning(self):
        f = friction_factor(1e5, D=0.1, eps=1e-4, fanning=True)
        fd = friction_factor(1e5, D=0.1, eps=1e-4, fanning=False)
        assert fd == pytest.approx(4 * f)

    def test_negative_re_raises(self):
        with pytest.raises(ValueError):
            friction_factor(-1.0, D=0.1, eps=1e-4)

    def test_vectorized_regimes(self):
        f = friction_factor([500, 3000, 1e5], D=0.1, eps=1e-4)
        assert f.shape == (3,)
        assert np.all(f > 0)


class TestSinglePhase:
    def test_horizontal_no_gravity(self):
        res = single_phase_gradient(10, 0.1, 1000, 1e-3)
        assert res.gravity == 0
        assert res.friction < 0
        assert res.total == pytest.approx(res.friction)

    def test_static_column(self):
        """Zero rate, vertical: pure hydrostatic gradient -rho*g."""
        res = single_phase_gradient(0, 0.1, 1000, 1e-3, inclination=1.0)
        assert res.friction == 0
        assert res.gravity == pytest.approx(-1000 * 9.80665, rel=1e-4)

    def test_reversed_flow_flips_friction_sign(self):
        fwd = single_phase_gradient(10, 0.1, 1000, 1e-3)
        rev = single_phase_gradient(-10, 0.1, 1000, 1e-3)
        assert rev.friction == pytest.approx(-fwd.friction)

    def test_supersonic_raises(self):
        with pytest.raises(ValueError, match="Supersonic"):
            single_phase_gradient(500, 0.05, 1.2, 1.8e-5, compressibility=1e-3)
