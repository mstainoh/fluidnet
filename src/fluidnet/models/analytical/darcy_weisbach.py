
from fluidnet.utils.fluid_functions import single_phase_pressure_gradient
from fluidnet.models.base import FlowHeadModel
from fluidnet.solvers import newton_inverse
import functools

__all__ = ['DarcyWeisbachModel']

inverse_pressure_gradient = functools.partial(
    newton_inverse, func=single_phase_pressure_gradient, low=1e-4, high=1)

class DarcyWeisbachModel(FlowHeadModel):
    def __init__(self, **default_params):
        self.default_params = dict(default_params)

    def head_difference_from_flow(self, rate, **kwargs):
        """
        Given flow rate and one head, compute the other.
        If both heads are None, return head difference.
        """
        dh = single_phase_pressure_gradient(rate, **self.default_params, **kwargs)
        return dh  # head drop

    def flow_from_heads(self, h_start, h_end=0, **kwargs):
        """
        Given head difference, compute flow rate.
        """
        dh = h_end - h_start
        return inverse_pressure_gradient(dh, **self.default_params, **kwargs)
