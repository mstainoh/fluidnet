from __future__ import annotations

from typing import ClassVar

from fluidnet.rate.base import CompositionalScalarRateBase


class BrineRate(CompositionalScalarRateBase):
    """Brine mass flow rate [kg/s]. Composition is a passive tracer (v0.2):
    it propagates and mixes, with no feedback on physical properties."""

    __slots__ = ()
    physics_key: ClassVar[str] = "mass_rate"
