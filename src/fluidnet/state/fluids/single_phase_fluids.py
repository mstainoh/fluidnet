"""Single-phase ``FluidState`` + ``Fluid`` implementations.

Layer zero, stateless (``CLAUDE.md`` #4): ``Fluid`` maps
``(composition, P, T) -> FluidState``, never freezes a density. Receives
composition as raw data, never a ``Rate``. Multi-phase counterparts live in
a sibling module (``multiphase_fluids.py``, not yet written).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import NamedTuple, cast

import numpy as np
import numpy.typing as npt

from fluidnet.physics.types import ArrayLike
from fluidnet.state.protocol import BoundStateModel


class SinglePhaseFluidState(NamedTuple):
    """Mono-phase ``FluidState``, bare field names (#19).

    Field names match the canonical kwargs of
    ``fluidnet.physics.single_phase.single_phase_gradient`` (#21) — the
    filtering-by-signature in ``loss_func`` depends on this.
    """

    density: ArrayLike
    viscosity: ArrayLike
    compressibility: ArrayLike

    def as_physics_kwargs(self) -> dict[str, ArrayLike]:
        return self._asdict()


class IncompressibleFluid:
    """Constant-property fluid — MVP baseline (ROADMAP Capa 0 bis, v0.2 scope).

    Ignores composition and temperature: properties are fixed at
    construction, not derived from an EOS. The bound state does not depend
    on ``x``/``across`` — the degenerate, scalar branch of #26.
    """

    def __init__(self, *, density: float, viscosity: float, compressibility: float = 0.0) -> None:
        self._state = SinglePhaseFluidState(
            density=density, viscosity=viscosity, compressibility=compressibility
        )

    def bind(
        self,
    ) -> BoundStateModel[SinglePhaseFluidState]:
        state = self._state

        def bound(*, x: float, across: ArrayLike) -> SinglePhaseFluidState:
            return state

        return bound


class CompressibleFluid(ABC):
    """Pressure-dependent fluid: density/viscosity/compressibility are an EOS
    evaluated per bound state, not fixed at construction (#4).

    Composition is bound as early as the case allows. A fluid whose EOS
    parameters are fixed for the network resolves them in ``__init__``
    (e.g. ``IdealGas(molar_weight=...)``) and its ``bind`` takes no
    composition at all. A compositional fluid (v1.5, composition comes from
    ``propagate_rates`` at runtime) overrides ``bind`` to accept it and
    resolves the EOS parameters once there, before returning the closure.
    Either way the hot loop never sees it: composition is constant along the
    edge in steady state with no mass exchange (#28). The base class does
    not know which case applies, so it declares neither.

    ``temperature`` in ``bind`` follows #26: ``None``/a scalar and a
    ``Callable[[float], float]`` are the two sibling cases, discriminated
    once in ``bind`` by ``callable(temperature)`` — no intermediate wrapper
    type. A callable is re-evaluated at ``x`` on every step (prescribed
    profile); ``None``/a scalar is captured once and ``x`` is ignored.
    """

    @abstractmethod
    def density(self, *, pressure: ArrayLike, temperature: float | None = None) -> ArrayLike:
        """Return density at given P, T (EOS)."""
        ...

    @abstractmethod
    def compressibility(
        self, *, pressure: ArrayLike, temperature: float | None = None
    ) -> ArrayLike:
        """Return isothermal compressibility at given P, T (EOS)."""
        ...

    @abstractmethod
    def viscosity(self, *, pressure: ArrayLike, temperature: float | None = None) -> ArrayLike:
        """Return viscosity at given P, T (EOS)."""
        ...

    def _state_at(self, *, pressure: ArrayLike, temperature: float | None) -> SinglePhaseFluidState:
        return SinglePhaseFluidState(
            density=self.density(pressure=pressure, temperature=temperature),
            viscosity=self.viscosity(pressure=pressure, temperature=temperature),
            compressibility=self.compressibility(pressure=pressure, temperature=temperature),
        )

    def bind(
        self,
        *,
        temperature: float | Callable[[float], float] | None = None,
    ) -> BoundStateModel[SinglePhaseFluidState]:
        if callable(temperature):
            profile = temperature

            def bound_callable(*, x: float, across: ArrayLike) -> SinglePhaseFluidState:
                P = cast(npt.NDArray[np.float64], across)[0]
                return self._state_at(pressure=P, temperature=profile(x))

            return bound_callable

        T = temperature

        def bound_fixed(*, x: float, across: ArrayLike) -> SinglePhaseFluidState:
            P = cast(npt.NDArray[np.float64], across)[0]
            return self._state_at(pressure=P, temperature=T)

        return bound_fixed
