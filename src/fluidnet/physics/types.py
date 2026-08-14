"""Physics-specific result containers. ``ArrayLike`` lives in ``fluidnet._types``."""

from typing import NamedTuple

from fluidnet._types import ArrayLike


class GradientResult(NamedTuple):
    """Pressure gradient decomposition [Pa/m], flow-direction sign convention.

    Supports tuple unpacking: ``total, dpg, dpf, dpv = result``.

    Attributes
    ----------
    gravity : ArrayLike
        Gravitational (hydrostatic) pressure gradient component [Pa/m].
    friction : ArrayLike
        Frictional pressure gradient component [Pa/m].
    momentum : ArrayLike
        Momentum (acceleration) pressure gradient component [Pa/m].
    """

    gravity: ArrayLike
    friction: ArrayLike
    momentum: ArrayLike

    @property
    def total(self) -> ArrayLike:
        """Total pressure gradient [Pa/m]."""
        return self.gravity + self.friction + self.momentum