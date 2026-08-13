"""``StateModel`` / ``BoundState`` — the neutral state protocol.

Domain-neutral by design (``CLAUDE.md`` #18): transforms the *across*
variable of a node (plus any prescribed fields) into whatever a
``gradient_fn`` needs. Fluids: ``(composition, T, P) -> (rho, mu, beta,
sigma)``. Electrical AC demo: ``(V) -> impedance``. Same shape either way.

Two methods, two binding times (#28, #30):

    StateModel.bind(**fields: object) -> BoundState   # 1x per edge
    BoundState.__call__(*, x, across) -> State         # 1x per step

``composition`` is not in this signature (correction 2026-08-13): not every
``StateModel`` has one (a fresh-water model can be parametrized by ``T``
alone). It is a field declared by the concrete implementation's ``bind``
instead (e.g. ``Fluid.bind(*, composition, temperature=None)``), same as
every other physical field (#18).

Architecture:
    ``StateModel`` is a factory defined at problem (network) level.
    ``BoundState`` is a callable defined at edge level.
    ``State`` is a concrete object defined at step (integration) level.
"""

from __future__ import annotations

from typing import Protocol

from fluidnet.physics.types import ArrayLike


class StateModel(Protocol):
    """Declarativo. Atributo de red (v1.0) o de nodo (v1.5)."""

    def bind(self, **fields: object) -> BoundState: ...


class BoundState(Protocol):
    """Ligado a un eje. Efímero: dura lo que dura la integración."""

    def __call__(self, *, x: float, across: ArrayLike) -> State: ...


class State(Protocol):
    def as_physics_kwargs(self) -> dict[str, ArrayLike]: ...
