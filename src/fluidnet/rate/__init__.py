"""Rate layer (``CLAUDE.md`` #3, #21, #22) — extensive quantity + intensive
composition, defined by contract (``Rate`` Protocol), not by content.

See ROADMAP.md v0.2 Scope IN and ``docs/design/architecture-v0.2.md``.
"""

from .base import BaseRate
from .fluids import BrineRate, MassRate, VolumetricRate
from .protocol import Rate

__all__ = ["Rate", "BaseRate", "MassRate", "VolumetricRate", "BrineRate"]
