"""Concrete ``Rate`` implementations for fluids (mass rate, volumetric rate,
brine). ``CLAUDE.md`` #33: concrete lives in a subpackage by domain,
mirroring ``state/protocol.py`` + ``state/fluids/``.
"""

from .brine import BrineRate
from .single_phase import MassRate, VolumetricRate

__all__ = ["MassRate", "VolumetricRate", "BrineRate"]
