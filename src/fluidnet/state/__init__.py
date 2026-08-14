"""State layer (layer zero, sibling of ``fluidnet.physics``).

Two pieces, deliberately separate:

``protocol``
    ``StateModel`` / ``BoundStateModel`` / ``State`` — the domain-neutral
    ``Protocol``. Knows nothing about fluids; the same shape fits an
    electrical AC demo.
``fluids``
    ``SinglePhaseFluidState``, ``IncompressibleFluid``,
    ``CompressibleFluidBase`` (+ concrete EOS ``IdealGas``/``RealGas``) —
    the fluid-specific implementation of that protocol. ``StateModel`` is
    the contract; ``CompressibleFluidBase`` is convenience for
    implementers, not a requirement (``CLAUDE.md`` #34).

See ``CLAUDE.md`` decisions #4-#7, #18, #19, #26, #28, #30, #34 and
``docs/design/architecture-v0.2.md`` §2.1bis.
"""

from .fluids import (
    CompressibleFluidBase,
    IdealGas,
    IncompressibleFluid,
    RealGas,
    SinglePhaseFluidState,
)
from .protocol import BoundStateModel, State, StateModel

__all__ = [
    "StateModel",
    "BoundStateModel",
    "State",
    "SinglePhaseFluidState",
    "IncompressibleFluid",
    "CompressibleFluidBase",
    "IdealGas",
    "RealGas",
]
