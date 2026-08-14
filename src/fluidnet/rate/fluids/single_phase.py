from __future__ import annotations

from typing import ClassVar

from fluidnet.rate.base import BaseRate


class MassRate(BaseRate):
    """Single-phase mass flow rate [kg/s], no composition."""

    __slots__ = ()
    physics_key: ClassVar[str] = "mass_rate"


class VolumetricRate(BaseRate):
    """Single-phase volumetric flow rate [m^3/s], no composition."""

    __slots__ = ()
    physics_key: ClassVar[str] = "volumetric_rate"
