"""Contract tests for the ``Rate`` protocol: ``MassRate`` and ``BrineRate``.

Mixing/validation/broadcasting behavior belongs to
``CompositionalScalarRateBase`` and is tested against a dummy in
``tests/test_rate_base.py``; this file keeps only what's specific to these
two concrete types.
"""

from __future__ import annotations

from functools import reduce
from operator import add

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


class TestBrineRateIdentity:
    def test_declares_mass_rate_physics_key(self) -> None:
        assert BrineRate.physics_key == "mass_rate"

    def test_inherits_weighted_mixing(self) -> None:
        mixed = BrineRate(1.0, {"NaCl": 0.0}) + BrineRate(3.0, {"NaCl": 0.2})
        assert mixed.value == 4.0
        assert mixed.composition["NaCl"] == pytest.approx(0.15)
