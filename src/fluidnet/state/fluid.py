"""``FluidState`` and ``Fluid`` — the fluid implementation of ``StateModel``.

Layer zero, stateless (``CLAUDE.md`` #4): ``Fluid`` is a factory that maps
``(composition, P, T) -> FluidState``, never an object with frozen density.
Receives composition as raw data (mapping/array), never a ``Rate`` (#4).

``FluidState`` is a ``NamedTuple`` of required fields, never ``float |
None`` (#5). Two sibling forms, phase-suffix convention (#19):
``SinglePhaseState`` (bare names) and ``MultiPhaseState`` (``_gas`` /
``_liquid`` suffixes), each with its own ``as_physics_kwargs()``.

Scope v0.2: ``IncompressibleFluid`` is the only concrete implementation
(ROADMAP, Capa 0 bis). ``IsothermalGas`` and compositional fluids land in
v1.0/v1.5.

Named ``fluid.py`` (singular, not ``fluids.py``) to avoid visually
colliding with ``fluids`` (ChEDL), the cross-validation oracle used in
tests (#16).
"""
