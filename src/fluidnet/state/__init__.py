"""State layer (layer zero, sibling of ``fluidnet.physics``).

Two pieces, deliberately separate:

``protocol``
    ``StateModel`` / ``BoundState`` / ``State`` — the domain-neutral
    ``Protocol``. Knows nothing about fluids; the same shape fits an
    electrical AC demo.
``fluids``
    ``SinglePhaseFluidState`` (+ future ``MultiPhaseFluidState``) and
    ``IncompressibleFluid`` (+ future ``IsothermalGas``) — the
    fluid-specific implementation of that protocol.

See ``CLAUDE.md`` decisions #4-#7, #18, #19, #26, #28, #30 and
``docs/design/architecture-v0.2.md`` §2.1bis.
"""

from .fluids import IncompressibleFluid, SinglePhaseFluidState
from .protocol import BoundState, State, StateModel

__all__ = [
    "StateModel",
    "BoundState",
    "State",
    "SinglePhaseFluidState",
    "IncompressibleFluid",
]
