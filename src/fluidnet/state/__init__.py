"""State layer (layer zero, sibling of ``fluidnet.physics``).

Two pieces, deliberately separate:

``protocol``
    ``StateModel`` / ``BoundState`` — the domain-neutral ``Protocol``. Knows
    nothing about fluids; the same shape fits an electrical AC demo.
``fluid``
    ``FluidState`` (+ ``SinglePhaseState`` / ``MultiPhaseState``) and
    ``Fluid`` / ``IncompressibleFluid`` — the fluid-specific implementation
    of that protocol.

See ``CLAUDE.md`` decisions #4-#7, #18, #19, #26, #28, #30 and
``docs/design/architecture-v0.2.md`` §2.1bis.
"""

# from .protocol import StateModel, BoundState
# from .fluid import FluidState, SinglePhaseState, MultiPhaseState, Fluid, IncompressibleFluid

__all__: list[str] = []
