"""Optional convenience base classes for ``Rate`` implementations.

Convenience, not requirement (``CLAUDE.md`` #34) — the library annotates
against the ``Rate`` Protocol, never against this class. ``BaseRate`` is an
abstract base providing the three network operations an implementor would
otherwise rewrite: mix at a node (``__add__``), scale by a split fraction
(``__mul__``), and flip sign by edge direction (``__neg__``). It declares no
``physics_key``/``physics_keys`` (#35); each subclass supplies its own.

Most rate quantities reduce to a float or a 1D array of floats, so the only
thing an implementation needs is the mapping from quantity to
the keyword the solver equation expects. That mapping is what the two
concrete siblings provide: ``ScalarBaseRate`` and ``VectorBaseRate``, which
share no inheritance between them since the scalar and vector
``physics_key`` conventions are incompatible ``ClassVar`` shapes.
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

    ``CompositionalScalarRateBase`` overrides ``_combine`` for subclasses
    that carry a passive-tracer composition — the extensive part still
    adds, the intensive part becomes a flow-weighted average.
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


class ScalarBaseRate(BaseRate):
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


class CompositionalScalarRateBase(ScalarBaseRate):
    """``ScalarBaseRate`` convenience for a scalar rate that also carries a
    passive-tracer composition (#22): it propagates and mixes with no
    feedback on physical properties. ``_combine`` sums the extensive part
    and flow-weights the composition; ``_rebuild`` leaves composition
    untouched, so ``__mul__``/``__neg__`` scale the extensive only.

    Precondition, not mechanism: ``value`` must be the extensive quantity
    that the composition is a fraction *of*, and that quantity must be
    conserved at the node — the mixing rule below only makes sense on that
    basis. Today this coincides with the quantity named by ``physics_key``
    (mass: it is conserved, and it is also what ``single_phase_gradient``
    consumes), but that's a coincidence of the current concrete subclasses,
    not a guarantee this base enforces — it is serving two distinct
    requirements with one ``value``: the mixing basis at propagation, and
    the kwarg the gradient equation consumes. A rate whose mixing basis is
    molar / standard-volume while its gradient equation consumes mass
    cannot use this base without a conversion that doesn't exist yet.
    """

    __slots__ = ("composition",)

    def __init__(self, value: ArrayLike, composition: dict[str, ArrayLike]) -> None:
        if np.any(np.asarray(value) < 0.0):
            raise ValueError(
                f"{type(self).__name__} requires a non-negative rate: a negative "
                "flow with positive composition has no physical meaning, and the "
                "flow-weighted mixing rule turns it into negative composition."
            )
        super().__init__(value)
        self.composition = composition

    def _rebuild(self, value: ArrayLike, composition: dict[str, ArrayLike] | None = None) -> Self:
        """Same intensive state, new extensive (#3: ``__mul__`` leaves
        composition intact). The optional parameter keeps the override
        compatible with ``BaseRate``'s one-argument call sites."""
        return type(self)(value, self.composition if composition is None else composition)

    def _combine(self, other: Self) -> Self:
        """#3: extensives add, composition is flow-weighted."""
        total = self.value + other.value
        # All-zero junction (every upstream well shut in): numerator is zero
        # too, so guarding the denominator yields 0.0. Guard the denominator
        # rather than np.where on the result — numpy would evaluate the
        # division anyway and emit RuntimeWarning.
        denom = np.where(total == 0.0, 1.0, total)
        mixed: dict[str, ArrayLike] = {
            k: (
                self.value * self.composition.get(k, 0.0)
                + other.value * other.composition.get(k, 0.0)
            )
            / denom
            for k in self.composition.keys() | other.composition.keys()
        }
        return self._rebuild(total, mixed)


class VectorBaseRate(BaseRate):
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
