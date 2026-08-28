"""``BaseRate`` — optional convenience base for ``Rate`` implementations.

Convenience, not requirement (``CLAUDE.md`` #34): the library annotates
against the ``Rate`` Protocol, never against this class. It exists so an
implementor gets the four network operations (mix at a node, scale by a
split fraction, flip sign by edge direction) without rewriting them.

``BaseRate`` itself declares no ``physics_key``/``physics_keys`` (#35): the
scalar and vector conventions are incompatible ``ClassVar`` shapes, so they
live on the two concrete siblings, ``ScalarRateBase`` and ``VectorRateBase``,
which share no inheritance between them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import numpy.typing as npt

from fluidnet._types import ArrayLike

if TYPE_CHECKING:
    from typing_extensions import Self


class BaseRate(ABC):
    """Single extensive quantity, no composition.

    Subclasses that carry composition (v1.5) override ``_combine`` — the
    extensive part still adds, the intensive part becomes a weighted
    average.
    """

    __slots__ = ("value",)

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
    @abstractmethod
    def as_physics_kwargs(self) -> dict[str, ArrayLike]: ...

    def __add__(self, other: Self) -> Self:
        if type(other) is not type(self):
            return NotImplemented
        return self._combine(other)

    # --- Utility algebra ----------------------------------------------
    def __mul__(self, scalar: ArrayLike) -> Self:
        return self._rebuild(self.value * scalar)

    __rmul__ = __mul__

    def __neg__(self) -> Self:
        return self._rebuild(-self.value)

    # --- Representation -----------------------------------------------
    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.value!r})"


class ScalarRateBase(BaseRate):
    """
    ``BaseRate`` convenience for a single named physics quantity (#35).
    
    Note that the value may be a scalar or an array. The equation itself
    is scalar, but the solver may evaluate it at multiple scenarios in parallel
    if the function allows vector input. 
    
    The physics key is always a single string.
    """

    __slots__ = ()

    #: Canonical ``physics`` kwarg name (#21). Class-level, never per-instance.
    physics_key: ClassVar[str]

    def as_physics_kwargs(self) -> dict[str, ArrayLike]:
        return {self.physics_key: self.value}


class VectorRateBase(BaseRate):
    """``BaseRate`` convenience for a packed multi-quantity rate (#35).

    ``value`` has shape ``(n_quantities, *scenario_shape)`` — quantity axis
    first, scenario axis last (#36). A split fraction of shape
    ``(n_quantities,)`` then broadcasts directly against ``value``: numpy
    aligns from the right, so putting the quantity axis last would
    broadcast the fraction against the scenario axis instead.
    """

    __slots__ = ()

    #: Canonical ``physics`` kwarg names, one per quantity axis entry (#21).
    physics_keys: ClassVar[tuple[str, ...]]

    #: Narrows ``BaseRate.value`` (#37 correction, 2026-08-28): a vector
    #: a vector rate should never be a bare scalar, 
    #  so unlike it doesn't need the ``float`` branch of ``ArrayLike``.
    value: npt.NDArray[np.float64]

    def __init__(self, value: npt.NDArray[np.float64]) -> None:
        value = np.asarray(value)
        if value.ndim == 0 or value.shape[0] != len(self.physics_keys):
            raise ValueError(
                f"{type(self).__name__} expects a leading quantity axis of "
                f"length {len(self.physics_keys)} (physics_keys), got shape "
                f"{value.shape}"
            )
        super().__init__(value)

    def as_physics_kwargs(self) -> dict[str, ArrayLike]:
        return dict(zip(self.physics_keys, self.value, strict=True))
