"""Array-shape / vectorization contract for `multiphase.beggs_brill`.

Tests the module's functions **in the order they are defined** (flowmap ->
holdup -> detailed -> public gradient), pinning what is vectorized today and
spec'ing what v0.5 should add. See `docs/design/physics-single-multiphase.md`
§3.3 and the roadmap item "Vectorización de `_beggs_brill_detailed`"
(ROADMAP.MD). `xfail(strict=True)` marks are executable spec, not skips: the
day broadcasting lands, these tests turn green and `strict=True` forces the
marker to be removed.
"""

import numpy as np
import pytest

from fluidnet.physics.multiphase import (
    FLOW_REGIMES,
    beggs_brill_flowmap,
    beggs_brill_gradient,
)
from fluidnet.physics.multiphase.beggs_brill import _beggs_brill_detailed, _holdup

# Shared scalar physical inputs (checalc.com sample, no Payne correction —
# same case as test_checalc_case_no_payne in test_beggs_brill_vs_book.py).
DETAILED_ARGS = dict(
    rho_liquid=613.8,
    rho_gas=141.3,
    mu_liquid=0.5e-3,
    mu_gas=0.02e-3,
    D=50e-3,
    inclination=1.0,
    roughness=0.0018e-3,
    sigma=28.0e-3,
    payne_correction=False,
)


class TestFlowmapVectorized:
    """`beggs_brill_flowmap` is vectorized today (boolean masks, no `if`)."""

    def test_flowmap_vectorized(self) -> None:
        """5 points spanning all 4 regimes, boundaries derived by hand from L1-L4."""
        Cl = np.array([0.005, 0.05, 0.05, 0.2, 0.5])
        NFr = np.array([10.0, 0.5, 4.0, 5.0, 100.0])
        expected_regimes = [
            "segregated", "segregated", "transition", "intermittent", "distributed",
        ]

        idx = beggs_brill_flowmap(Cl, NFr)

        assert idx.shape == (5,)
        assert [FLOW_REGIMES[i] for i in idx] == expected_regimes

    def test_flowmap_rejects_out_of_domain(self) -> None:
        """NaN falsifies every boundary mask -> ValueError, not silent NaN passthrough.

        ``Cl=0`` is *not* a valid probe for this: it falls cleanly into the
        distributed mask instead of landing outside all four regimes.
        """
        Cl = np.array([0.2, np.nan])
        NFr = np.array([5.0, 5.0])
        with pytest.raises(ValueError):
            beggs_brill_flowmap(Cl, NFr)


class TestHoldupVectorized:
    """`_holdup` vectorizes over Cl/NFr/Nlv for a *fixed* regime index ``i``.

    ``angle`` stays scalar by design: one pipe inclination per call, not one
    angle per data point.
    """

    @pytest.mark.parametrize("i", [0, 1, 2, 3])
    def test_holdup_vectorized_within_regime(self, i: int) -> None:
        Cl = np.array([0.05, 0.1, 0.2])
        NFr = np.array([4.0, 5.0, 6.0])
        Nlv = np.array([1.0, 1.2, 1.5])

        el = _holdup(i, Cl, NFr, Nlv, angle=0.3)

        assert np.asarray(el).shape == (3,)
        assert np.all(np.isfinite(el))


class TestDetailedScalarContract:
    """`_beggs_brill_detailed` is scalar-only today (documented, not a bug —
    see CLAUDE.md and physics-single-multiphase.md §"Contrato de forma").
    """

    def test_detailed_scalar_contract_today(self) -> None:
        """Scalars in -> scalars out; no stray 0-d array leaks into GradientResult."""
        calc = _beggs_brill_detailed(
            liquid_mass_rate=0.8, gas_mass_rate=0.05, **DETAILED_ARGS
        )
        grad = calc["gradient"]

        for value in (
            grad.total, grad.gravity, grad.friction, grad.momentum,
            calc["NFr"], calc["liquid_holdup"],
        ):
            assert np.ndim(value) == 0
            assert not isinstance(value, np.ndarray)

    def test_detailed_rejects_array_rates(self) -> None:
        """Documents *why* it's scalar-only: the co/counter-flow check does
        `if liquid_mass_rate > 0 and ...`, which is ambiguous for arrays."""
        ql = np.array([0.8, 1.0])
        qg = np.array([0.05, 0.06])
        with pytest.raises(ValueError, match="ambiguous"):
            _beggs_brill_detailed(liquid_mass_rate=ql, gas_mass_rate=qg, **DETAILED_ARGS)

    @pytest.mark.xfail(strict=True, reason="v0.5 roadmap: broadcast rates -> vectorized GradientResult")
    def test_detailed_vectorized_over_rates(self) -> None:
        ql = np.array([0.8, 1.0, 1.2])
        qg = np.array([0.05, 0.06, 0.07])
        calc = _beggs_brill_detailed(liquid_mass_rate=ql, gas_mass_rate=qg, **DETAILED_ARGS)
        assert calc["gradient"].total.shape == (3,)


class TestGradientScalarContract:
    """Same scalar-only contract at the public `beggs_brill_gradient` entry
    point — it wraps `_beggs_brill_detailed` directly, so it inherits both
    the limitation and (eventually) the fix."""

    def test_gradient_scalar_contract_today(self) -> None:
        grad = beggs_brill_gradient(liquid_mass_rate=0.8, gas_mass_rate=0.05, **DETAILED_ARGS)

        for value in grad:
            assert np.ndim(value) == 0
            assert not isinstance(value, np.ndarray)

    def test_gradient_rejects_array_rates(self) -> None:
        ql = np.array([0.8, 1.0])
        qg = np.array([0.05, 0.06])
        with pytest.raises(ValueError, match="ambiguous"):
            beggs_brill_gradient(liquid_mass_rate=ql, gas_mass_rate=qg, **DETAILED_ARGS)

    @pytest.mark.xfail(strict=True, reason="v0.5 roadmap: broadcast rates -> vectorized GradientResult")
    def test_gradient_vectorized_over_rates(self) -> None:
        ql = np.array([0.8, 1.0, 1.2])
        qg = np.array([0.05, 0.06, 0.07])
        grad = beggs_brill_gradient(liquid_mass_rate=ql, gas_mass_rate=qg, **DETAILED_ARGS)
        assert grad.total.shape == (3,)
