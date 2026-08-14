"""Cross-cutting type aliases (layer -1: imports nothing from the package).

Used by ``physics/``, ``state/``, and eventually ``rate/`` and ``losses/``.
"""

from typing import TypeAlias

import numpy as np
import numpy.typing as npt

#: A physics quantity: either a Python float or a numpy array of floats.
#: Every pure physics function in this package is ``SI -> SI`` and
#: vectorized, so inputs/outputs are typed with this alias throughout.
ArrayLike: TypeAlias = float | npt.NDArray[np.float64]
