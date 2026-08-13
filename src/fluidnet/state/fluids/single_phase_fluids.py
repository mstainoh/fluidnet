"""Single-phase ``FluidState`` + ``Fluid`` implementations.

Layer zero, stateless (``CLAUDE.md`` #4): ``Fluid`` maps
``(composition, P, T) -> FluidState``, never freezes a density. Receives
composition as raw data, never a ``Rate``. Multi-phase counterparts live in
a sibling module (``multiphase_fluids.py``, not yet written).
"""

from __future__ import annotations

from typing import NamedTuple

from fluidnet.physics.types import ArrayLike
from fluidnet.state.protocol import BoundState


class SinglePhaseFluidState(NamedTuple):
    """Mono-phase ``FluidState``, bare field names (#19).

    Field names match the canonical kwargs of
    ``fluidnet.physics.single_phase.single_phase_gradient`` (#21) — the
    filtering-by-signature in ``loss_func`` depends on this.
    """

    density: float
    viscosity: float
    compressibility: float

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
        self, *, composition: dict[str, float] | None = None, temperature: float | None = None
    ) -> BoundState:
        state = self._state

        def bound(*, x: float, across: ArrayLike) -> SinglePhaseFluidState:
            return state

        return bound
