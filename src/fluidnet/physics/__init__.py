"""Pure physics functions (layer zero).

All functions in this package are pure ``SI -> SI``: no units handling, no
graph/network knowledge, no global state. They are wrapped later by
``fluidnet.losses`` to satisfy the ``AlgebraicLoss`` / ``IntegralLoss``
protocols.

Sign convention (package-wide)
------------------------------
Gradients are expressed *in the direction of positive flow*, e.g.:
``dp = p_downstream - p_upstream``. A **loss** is therefore **negative**
for positive rates, and positive for negative (reversed) rates.

Note that many correlations / functions are expressed taking the flow quantity
as strictly non-negative (e.g., for dimensionless numbers).
"""

from .dimensionless import froude, mach, reynolds
from .friction import friction_factor
from .multiphase import beggs_brill_gradient
from .single_phase import single_phase_gradient
from .types import GradientResult

__all__ = [
    "reynolds",
    "froude",
    "mach",
    "friction_factor",
    "single_phase_gradient",
    "beggs_brill_gradient",
    "GradientResult",
]
