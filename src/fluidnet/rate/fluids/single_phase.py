"""Standard ``Rate`` implementations for single-phase fluids.

No composition, no phase vector. Each declares the ``physics_key`` that
``as_physics_kwargs()`` uses to hoist it into the gradient kwargs (#21, #22).

``MassRate``: passes values as ``mass_rate``.
``VolumetricRate``: passes values as ``volumetric_rate``.
"""

from __future__ import annotations

from typing import ClassVar

from fluidnet.rate.base import ScalarRateBase


class MassRate(ScalarRateBase):
    """Single-phase mass flow rate [kg/s], no composition."""

    __slots__ = ()
    physics_key: ClassVar[str] = "mass_rate"


class VolumetricRate(ScalarRateBase):
    """Single-phase volumetric flow rate [m^3/s], no composition."""

    __slots__ = ()
    physics_key: ClassVar[str] = "volumetric_rate"
