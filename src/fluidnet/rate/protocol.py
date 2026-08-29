"""``Rate`` — the through-variable protocol.

A ``Rate`` expresses the flow of an extensive quantity through the network:
a scalar mass rate, a vector of phase rates, a scalar plus composition
(``BrineRate``), or any topologically equivalent quantity — one that adds
across nodes the way flow does.

Two things happen to it during a solve. Before integration,
``as_physics_kwargs()`` unpacks it into the flat SI kwargs ``gradient_fn``
needs, hoisted once per edge (``CLAUDE.md`` #22) — no ``Rate`` object enters
the integration loop itself. After integration, ``propagate_rates`` walks
the solved network and accumulates rates at nodes via ``__add__``; node mass
balance, when it applies, is simply ``reduce(add, rates)`` of in vs. out
edges. The contract is therefore exactly two methods: ``as_physics_kwargs``
and ``__add__``.

Note: ``physics_key`` (classvar) is not part of the protocol (``CLAUDE.md`` #21). 
Reason: The canonical physics vocabulary lives in the *keys of the dict* 
``as_physics_kwargs()``returns, not in a member of this Protocol. ``physics_key: str`` /
``physics_keys: tuple[str, ...]`` are the mechanism the convenience bases
``ScalarBaseRate``/``VectorBaseRate`` use to *implement* that method — a
``ClassVar`` internal to those classes. Declaring the scalar spelling here
would also exclude every vector rate from the Protocol, since its
``ClassVar`` is the plural tuple; an implementation that builds the dict by
hand, with no ``physics_key`` at all, satisfies this contract equally.

Scaling (``__mul__``/``__neg__``) used to live on this Protocol, from when
``Rate`` objects were expected to flow through solver machinery directly.
Now that the solver only ever touches the unpacked kwargs, scaling isn't a
generic requirement — it stays a concrete convenience on ``BaseRate`` (for
the v0.5 fitting optimizer, see ROADMAP) rather than on the minimal
Protocol.

Generic via ``Self`` (correction 2026-08-14): parameters are contravariant,
so a protocol promising ``other: Rate`` would force every implementation to
accept any ``Rate`` — and ``BrineRate.__add__(other: BrineRate)`` would stop
satisfying it under ``--strict``. ``Self`` makes the contract *"my own
type"*, checked statically instead of deferred to a runtime
``NotImplemented``.

No ``__radd__``: node balance is ``reduce(add, rates)``, which needs no
seed. No ``__sub__``: it has no client and invites ``rate_in - rate_out``,
the pattern that cancels the mixing denominator.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from fluidnet._types import ArrayLike

if TYPE_CHECKING:  # Self is 3.11+; the package floor is 3.10
    from typing_extensions import Self


class Rate(Protocol):
    """A through-variable: conserved at nodes, propagated through the network."""

    def as_physics_kwargs(self) -> dict[str, ArrayLike]:
        """Extensive quantity, canonical name, SI. Hoisted per edge (#22)."""
        ...

    def __add__(self, other: Self) -> Self: ...
