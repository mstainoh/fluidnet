from fluidnet.models.base import FlowHeadModel
from scipy.integrate import solve_ivp

class NumericalGradientModel(FlowHeadModel):
    method = 'RK45'
    max_steps = 100

    def gradient_function(self, x, h, rate, **kwargs):
        # override in subclass
        raise NotImplementedError

    def integrate_gradient_function(self, a, b, rate, full_output=False, **kwargs):
        result = solve_ivp(
            lambda x, h: self.gradient_function(x, h, rate, **{**self.default_params, **kwargs}),
            t_span=[a, b],
            y0=[0],
            method='RK45', max_step=(b - a)/self.max_steps
        )
        if full_output:
            return result
        return result.y[0][-1]
    
    
    def start_head_from_end_and_flow(self, rate, h_end=None, full_output=False, **kwargs):
        h_end = h_end or 0
        return self.integrate_gradient_function()

    def head_difference_from_flow(self, rate, L=1, steps=100, direction="forward", **kwargs):
        if direction == "backward":
            x_span = [L, 0]
        else:
            x_span = [0, L]
        result = solve_ivp(
            lambda x, h: self.gradient_function(x, h, rate, **{**self.default_params, **kwargs}),
            t_span=x_span,
            y0=[0],
            method='RK45', max_step=L/steps
        )
        return result.y[0][-1]

    def flow_from_heads(self, h_start, h_end, **kwargs):
        raise NotImplementedError("Reverse computation for flow rate not implemented numerically.")

class SinglePhaseModel:
    def __init__(self):
