"""``StateModel`` / ``BoundStateModel`` — the neutral state protocol.

Domain-neutral by design (``CLAUDE.md`` #18): transforms the *across*
variable of a node (plus any prescribed fields) into whatever a
``gradient_fn`` needs. Fluids: ``(composition, T, P) -> (rho, mu, beta,
sigma)``. Electrical AC demo: ``(V) -> impedance``. Same shape either way.

Two methods, two binding times (#28, #30):

    StateModel.bind(**fields: object) -> BoundStateModel   # 1x per edge
    BoundStateModel.__call__(*, x, across) -> State         # 1x per step

``composition`` is not in this signature (correction 2026-08-13): not every
``StateModel`` has one (a fresh-water model can be parametrized by ``T``
alone). It is a field declared by the concrete implementation's ``bind``
instead (e.g. ``Fluid.bind(*, composition, temperature=None)``), same as
every other physical field (#18).

**Generic over the concrete ``State`` (2026-08-13).** Both ``StateModel``
and ``BoundStateModel`` carry a covariant type parameter bound to ``State``.
Without it, every concrete ``bind()`` had to widen its return type to the
bare ``BoundStateModel``, whose ``__call__`` returns the neutral ``State``
Protocol — callers lost the concrete shape (``state.density`` would not
type-check, only ``state.as_physics_kwargs()``) even though the object
returned at runtime always was the concrete subtype. A concrete
implementation now types its ``bind`` as
``bind(...) -> BoundStateModel[SinglePhaseFluidState]``, and the concrete
field names survive the whole ``bind → BoundStateModel → State`` chain.
``StateModel``/``BoundStateModel`` stay neutral (the type parameter is not
bound to any fluid-specific type here); only concrete implementations name
a concrete ``State`` subtype.

Architecture:
    ``StateModel`` is a factory defined at problem (network) level.
    ``BoundStateModel`` is a callable defined at edge level.
    ``State`` is a concrete object defined at step (integration) level.
"""

from __future__ import annotations

from typing import Protocol, TypeVar

from fluidnet._types import ArrayLike

S_co = TypeVar("S_co", bound="State", covariant=True)


class StateModel(Protocol[S_co]):
    """Declarativo. Atributo de red (v1.0) o de nodo (v1.5)."""

    def bind(self, **fields: object) -> BoundStateModel[S_co]: ...


class BoundStateModel(Protocol[S_co]):
    """Ligado a un eje. Efímero: dura lo que dura la integración."""

    def __call__(self, *, x: float, across: ArrayLike) -> S_co: ...


class State(Protocol):
    def as_physics_kwargs(self) -> dict[str, ArrayLike]: ...
