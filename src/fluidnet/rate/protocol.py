"""``Rate`` — the through-variable protocol.

Contract, not content (``CLAUDE.md`` #22): *I can be summed with others of
my own kind to give zero balance, and I can enter a loss function.*

Generic via ``Self`` (correction 2026-08-14): parameters are contravariant,
so a protocol promising ``other: Rate`` would force every implementation to
accept any ``Rate`` — and ``BrineRate.__add__(other: BrineRate)`` would stop
satisfying it under ``--strict``. ``Self`` makes the contract *"my own
type"*, checked statically instead of deferred to a runtime
``NotImplemented``.

No ``__radd__``: node balance is ``reduce(add, rates)``, which needs no
seed. ``sum()`` would start at ``0`` and force a ``Literal[0]`` into the
signature. No ``__sub__``: it has no client and invites
``rate_in - rate_out``, the pattern that cancels the mixing denominator.

The solver does not consume this algebra. It balances raw extensives
(``.value``) and ``as_physics_kwargs`` is hoisted once per edge (#22), so
no ``Rate`` object enters the integration loop. ``__add__`` belongs to
``propagate_rates`` — a topological pass over the already-solved network,
decoupled from the solve.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Protocol

from fluidnet._types import ArrayLike

if TYPE_CHECKING:  # Self is 3.11+; the package floor is 3.10
    from typing_extensions import Self


class Rate(Protocol):
    """A through-variable: conserved at nodes, scalable along edges."""

    #: Canonical ``physics`` kwarg name for the extensive quantity (#21).
    #: Class-level: the name belongs to the type, never to the instance.
    physics_key: ClassVar[str]

    def as_physics_kwargs(self) -> dict[str, ArrayLike]:
        """Extensive quantity, canonical name, SI. Hoisted per edge (#22)."""
        ...

    def __add__(self, other: Self) -> Self: ...
    def __mul__(self, scalar: ArrayLike) -> Self: ...
    def __rmul__(self, scalar: ArrayLike) -> Self: ...
    def __neg__(self) -> Self: ...
