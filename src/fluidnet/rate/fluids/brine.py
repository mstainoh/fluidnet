from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import numpy as np

from fluidnet._types import ArrayLike
from fluidnet.rate.base import ScalarBaseRate

if TYPE_CHECKING:
    from typing_extensions import Self


class BrineRate(ScalarBaseRate):
    """Brine mass flow rate [kg/s]. Composition is a passive tracer (v0.2):
    it propagates and mixes, with no feedback on physical properties."""

    __slots__ = ("composition",)
    physics_key: ClassVar[str] = "mass_rate"

    def __init__(self, value: ArrayLike, composition: dict[str, ArrayLike]) -> None:
        if np.any(np.asarray(value) < 0.0):
            raise ValueError(
                "BrineRate requires a non-negative mass rate: a negative flow "
                "with positive composition has no physical meaning, and the "
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
