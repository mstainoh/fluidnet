"""Shared type aliases and result containers for the physics layer."""

from typing import NamedTuple, TypeAlias

import numpy as np
import numpy.typing as npt

#: A physics quantity: either a Python float or a numpy array of floats.
#: Every pure physics function in this package is ``SI -> SI`` and
#: vectorized, so inputs/outputs are typed with this alias throughout.
ArrayLike: TypeAlias = float | npt.NDArray[np.float64]


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