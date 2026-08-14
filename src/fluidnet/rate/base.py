"""``BaseRate`` — optional convenience base for ``Rate`` implementations.

Convenience, not requirement (``CLAUDE.md`` #34): the library annotates
against the ``Rate`` Protocol, never against this class. It exists so an
implementor gets the four network operations (mix at a node, scale by a
split fraction, flip sign by edge direction) without rewriting them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from fluidnet._types import ArrayLike

if TYPE_CHECKING:
    from typing_extensions import Self


class BaseRate:
    """Single extensive quantity, no composition.

    Subclasses set ``physics_key`` and nothing else. Subclasses that carry
    composition (v1.5) override ``_combine`` — the extensive part still
    adds, the intensive part becomes a weighted average.
    """

    __slots__ = ("value",)

    #: Canonical ``physics`` kwarg name (#21). Class-level, never per-instance.
    physics_key: ClassVar[str]

    #: Keep numpy from hijacking ``ndarray * rate`` into an object array.
    __array_ufunc__: ClassVar[None] = None

    def __init__(self, value: ArrayLike) -> None:
        self.value = value

    # --- construction hooks -------------------------------------------
    def _rebuild(self, value: ArrayLike) -> Self:
        """Build a sibling carrying ``value``. Override if ``__init__`` grows."""
        return type(self)(value)

    def _combine(self, other: Self) -> Self:
        """Mixing rule. Pure addition here; weighted average once composition exists."""
        return self._rebuild(self.value + other.value)

    # --- Rate protocol ------------------------------------------------
    def as_physics_kwargs(self) -> dict[str, ArrayLike]:
        return {self.physics_key: self.value}

    def __add__(self, other: Self) -> Self:
        if type(other) is not type(self):
            return NotImplemented
        return self._combine(other)

    def __mul__(self, scalar: ArrayLike) -> Self:
        return self._rebuild(self.value * scalar)

    __rmul__ = __mul__

    def __neg__(self) -> Self:
        return self._rebuild(-self.value)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.value!r})"
