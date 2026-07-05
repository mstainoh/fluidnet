"""Pure physics functions (layer zero).

All functions in this package are pure ``SI -> SI``: no units handling, no
graph/network knowledge, no global state. They are wrapped later by
``fluidnet.losses`` to satisfy the ``AlgebraicLoss`` / ``IntegralLoss``
protocols.

Sign convention (package-wide)
------------------------------
Pressure gradients are expressed *in the direction of positive flow*:
``dp = p_downstream - p_upstream``. A pressure **loss** is therefore
**negative** for positive rates, and positive for negative (reversed) rates.
"""

from .dimensionless import reynolds, froude, mach
from .friction import friction_factor, ROUGHNESS_VALUES
from .single_phase import single_phase_gradient, GradientResult
from .multiphase import beggs_brill_gradient

__all__ = [
    "reynolds",
    "froude",
    "mach",
    "friction_factor",
    "ROUGHNESS_VALUES",
    "single_phase_gradient",
    "beggs_brill_gradient",
    "GradientResult",
]
