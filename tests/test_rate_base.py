"""Contract tests for ``BaseRate``/``ScalarBaseRate``/``VectorBaseRate``
(``CLAUDE.md`` #35–#37).

Dummy subclasses defined here, not in the package (``VectorBaseRate`` has no
concrete subclass in ``fluidnet.rate`` yet): ``physics_keys = ("q_a", "q_b")``
are deliberately dummy — the point is the contract, not the physics.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pytest

from fluidnet.rate.base import ScalarBaseRate, VectorBaseRate


class _ScalarTestRate(ScalarBaseRate):
    __slots__ = ()
    physics_key: ClassVar[str] = "q"


class _TwoPhaseTestRate(VectorBaseRate):
    __slots__ = ()
    physics_keys: ClassVar[tuple[str, ...]] = ("q_a", "q_b")


class TestScalarAsPhysicsKwargs:
    def test_maps_physics_key_to_value(self) -> None:
        assert _ScalarTestRate(10.0).as_physics_kwargs() == {"q": 10.0}


class TestVectorAsPhysicsKwargs:
    def test_key_zero_binds_to_row_zero_not_inverted(self) -> None:
        """Guards against the zip(value, physics_keys) bug: the previous
        version bound the key to the numeric value instead of the row."""
        rate = _TwoPhaseTestRate(np.array([[1.0, 2.0], [10.0, 20.0]]))
        kwargs = rate.as_physics_kwargs()
        assert set(kwargs) == {"q_a", "q_b"}
        assert np.array_equal(kwargs["q_a"], np.array([1.0, 2.0]))
        assert np.array_equal(kwargs["q_b"], np.array([10.0, 20.0]))


class TestVectorConstructionValidation:
    def test_scalar_input_raises(self) -> None:
        with pytest.raises(ValueError):
            _TwoPhaseTestRate(5.0)  # type: ignore[arg-type]

    def test_wrong_leading_axis_length_raises(self) -> None:
        with pytest.raises(ValueError):
            _TwoPhaseTestRate(np.zeros((3, 5)))


class TestQuantityAxisFirstBroadcasting:
    def test_quantity_axis_first_broadcasts_split_fraction_correctly(self) -> None:
        """#36: value is (n_quantities, *scenario_shape), so a split fraction
        of shape (scenario,) broadcasts against the scenario axis, not the
        quantity axis."""
        value = np.arange(10.0).reshape(2, 5)
        split = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        scaled = _TwoPhaseTestRate(value) * split
        assert scaled.value.shape == (2, 5)
        assert np.allclose(scaled.value, value * split)


class TestCrossTypeAddition:
    def test_add_returns_not_implemented_directly(self) -> None:
        result = _ScalarTestRate(1.0).__add__(_TwoPhaseTestRate(np.array([1.0, 2.0])))  # type: ignore[operator]
        assert result is NotImplemented

    def test_scalar_plus_vector_raises_type_error(self) -> None:
        with pytest.raises(TypeError):
            _ScalarTestRate(1.0) + _TwoPhaseTestRate(np.array([1.0, 2.0]))  # type: ignore[operator]

    def test_vector_plus_scalar_raises_type_error(self) -> None:
        with pytest.raises(TypeError):
            _TwoPhaseTestRate(np.array([1.0, 2.0])) + _ScalarTestRate(1.0)  # type: ignore[operator]


class TestArrayUfuncOverride:
    def test_negation_preserves_type(self) -> None:
        neg = -_ScalarTestRate(3.0)
        assert isinstance(neg, _ScalarTestRate)
        assert neg.value == -3.0

    def test_ndarray_on_left_delegates_to_rmul(self) -> None:
        """__array_ufunc__ = None keeps numpy from hijacking ndarray * rate
        into an object array; it defers to Rate.__rmul__ instead."""
        scaled = np.array(2.0) * _ScalarTestRate(3.0)
        assert isinstance(scaled, _ScalarTestRate)
        assert scaled.value == 6.0


class TestRebuildPreservesConcreteType:
    def test_scalar_mul_neg_add(self) -> None:
        rate = _ScalarTestRate(2.0)
        assert isinstance(rate * 3, _ScalarTestRate)
        assert isinstance(-rate, _ScalarTestRate)
        assert isinstance(rate + _ScalarTestRate(1.0), _ScalarTestRate)

    def test_vector_mul_neg_add(self) -> None:
        rate = _TwoPhaseTestRate(np.array([1.0, 2.0]))
        assert isinstance(rate * 3, _TwoPhaseTestRate)
        assert isinstance(-rate, _TwoPhaseTestRate)
        assert isinstance(rate + _TwoPhaseTestRate(np.array([1.0, 2.0])), _TwoPhaseTestRate)
