"""Tests for SinglePhaseFluidState and IncompressibleFluid."""

import pytest

from fluidnet.physics.single_phase import single_phase_gradient
from fluidnet.state.fluids.single_phase import IncompressibleFluid, SinglePhaseFluidState


class TestSinglePhaseFluidState:
    def test_construction_by_keyword(self) -> None:
        state = SinglePhaseFluidState(density=1000.0, viscosity=1e-3, compressibility=0.0)
        assert state.density == 1000.0
        assert state.viscosity == 1e-3
        assert state.compressibility == 0.0

    def test_immutable(self) -> None:
        state = SinglePhaseFluidState(density=1000.0, viscosity=1e-3, compressibility=0.0)
        with pytest.raises(AttributeError):
            state.density = 999.0  # type: ignore[misc]

    def test_as_physics_kwargs_matches_canonical_names(self) -> None:
        """Field names must match single_phase_gradient's kwargs verbatim (#21):
        no rename table, loss_func composes by signature filtering alone."""
        state = SinglePhaseFluidState(density=1000.0, viscosity=1e-3, compressibility=0.0)
        assert set(state.as_physics_kwargs()) == {"density", "viscosity", "compressibility"}

    def test_as_physics_kwargs_feeds_gradient_fn(self) -> None:
        state = SinglePhaseFluidState(density=1000.0, viscosity=1e-3, compressibility=0.0)
        result = single_phase_gradient(mass_rate=10.0, D=0.1, **state.as_physics_kwargs())
        assert result.friction < 0


class TestIncompressibleFluid:
    def test_compressibility_defaults_to_zero(self) -> None:
        fluid = IncompressibleFluid(density=1000.0, viscosity=1e-3)
        state = fluid.bind()(x=0.0, across=1e5)
        assert state.compressibility == 0.0

    def test_bind_takes_no_composition_or_temperature(self) -> None:
        """Properties are fixed at construction (#28): IncompressibleFluid
        uses neither field, so bind() does not declare them at all — passing
        either is a TypeError, not a silent no-op."""
        fluid = IncompressibleFluid(density=1000.0, viscosity=1e-3)
        with pytest.raises(TypeError):
            fluid.bind(composition={"water": 1.0})  # type: ignore[call-arg]
        with pytest.raises(TypeError):
            fluid.bind(temperature=350.0)  # type: ignore[call-arg]

    def test_bound_state_ignores_x_and_across(self) -> None:
        """Degenerate scalar branch of #26: the bound state does not depend
        on x or across, since properties are fixed at construction."""
        bound = IncompressibleFluid(density=1000.0, viscosity=1e-3).bind()
        assert bound(x=0.0, across=1e5) == bound(x=50.0, across=9e5)

    def test_bind_is_pure_repeated_binds_are_independent(self) -> None:
        """bind() is partial application, not construction (#18): it must not
        mutate the Fluid, so independent binds/evaluations never interfere."""
        fluid = IncompressibleFluid(density=1000.0, viscosity=1e-3)
        bound_a = fluid.bind()
        bound_a(x=0.0, across=1e5)
        bound_b = fluid.bind()
        assert bound_b(x=0.0, across=1e5) == bound_a(x=99.0, across=9e5)

    def test_bind_takes_no_positional_args(self) -> None:
        """bind() declares zero fields (#28) — no composition/temperature to
        be kw-only about anymore, but a positional call must still fail."""
        fluid = IncompressibleFluid(density=1000.0, viscosity=1e-3)
        with pytest.raises(TypeError):
            fluid.bind({"water": 1.0})  # type: ignore[call-arg]

    def test_end_to_end_single_phase_gradient(self) -> None:
        fluid = IncompressibleFluid(density=1000.0, viscosity=1e-3, compressibility=0.0)
        state = fluid.bind()(x=0.0, across=5e5)
        result = single_phase_gradient(mass_rate=10.0, D=0.1, **state.as_physics_kwargs())
        assert result.total < 0
