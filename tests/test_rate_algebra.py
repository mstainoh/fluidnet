"""Contract tests for the ``Rate`` protocol: ``MassRate`` and ``BrineRate``.

No ``__eq__`` on ``BaseRate`` (CLAUDE.md #22), so assertions compare
``.value``/``.composition`` directly (``np.allclose`` for array values)
instead of comparing ``Rate`` instances.
"""

from __future__ import annotations

import warnings
from functools import reduce
from operator import add

import numpy as np
import pytest

from fluidnet.rate import BrineRate, MassRate


class TestMassRateAlgebra:
    def test_as_physics_kwargs_uses_canonical_name(self) -> None:
        assert MassRate(10.0).as_physics_kwargs() == {"mass_rate": 10.0}

    def test_node_balance_via_reduce(self) -> None:
        """No __radd__/sum() (#22): node balance is reduce(add, rates)."""
        total = reduce(add, [MassRate(2.0), MassRate(3.0)])
        assert total.value == 5.0

    def test_add_across_types_raises_type_error(self) -> None:
        """__add__ returns NotImplemented for a foreign type (#22, Self);
        with no __radd__ on either side, Python raises TypeError."""
        with pytest.raises(TypeError):
            MassRate(1.0) + BrineRate(1.0, {"NaCl": 0.1})  # type: ignore[operator]


class TestBrineRateScaling:
    def test_scaling_preserves_composition(self) -> None:
        """#3: __mul__ scales the extensive, composition is invariant."""
        scaled = BrineRate(5.0, {"NaCl": 0.1}) * 3
        assert scaled.value == 15.0
        assert scaled.composition == {"NaCl": 0.1}


class TestBrineRateMixing:
    def test_weighted_mixing(self) -> None:
        mixed = BrineRate(1.0, {"NaCl": 0.0}) + BrineRate(3.0, {"NaCl": 0.2})
        assert mixed.value == 4.0
        assert np.isclose(mixed.composition["NaCl"], 0.15)

    def test_key_union_dilutes_species_absent_from_one_side(self) -> None:
        mixed = BrineRate(1.0, {"NaCl": 0.5}) + BrineRate(1.0, {})
        assert mixed.value == 2.0
        assert np.isclose(mixed.composition["NaCl"], 0.25)

    def test_zero_contribution_no_warning(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            mixed = BrineRate(0.0, {}) + BrineRate(5.0, {"NaCl": 0.1})
        assert mixed.value == 5.0
        assert mixed.composition == {"NaCl": 0.1}

    def test_all_zero_junction_no_error(self) -> None:
        """Every upstream well shut in: guarded denominator yields 0.0,
        not ZeroDivisionError/RuntimeWarning (ROADMAP §Abiertas, guard is
        provisional to DAG + non-negative rates)."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            mixed = BrineRate(0.0, {}) + BrineRate(0.0, {})
        assert mixed.value == 0.0


class TestBrineRateValidation:
    def test_negative_scalar_raises(self) -> None:
        with pytest.raises(ValueError):
            BrineRate(-1.0, {})

    def test_negative_array_element_raises(self) -> None:
        with pytest.raises(ValueError):
            BrineRate(np.array([1.0, -1.0]), {})


class TestBrineRateBroadcasting:
    def test_mixes_against_scalar_valued_rate(self) -> None:
        vector_rate = BrineRate(np.array([1.0, 3.0]), {"NaCl": 0.1})
        scalar_rate = BrineRate(2.0, {"NaCl": 0.1})
        mixed = vector_rate + scalar_rate
        assert np.allclose(mixed.value, np.array([3.0, 5.0]))
        assert np.allclose(mixed.composition["NaCl"], 0.1)

    def test_incompatible_shapes_raise(self) -> None:
        a = BrineRate(np.array([1.0, 3.0]), {"NaCl": 0.1})
        b = BrineRate(np.array([1.0, 2.0, 3.0]), {"NaCl": 0.1})
        with pytest.raises(ValueError):
            a + b
