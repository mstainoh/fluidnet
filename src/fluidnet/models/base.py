
from abc import ABC, abstractmethod

class FlowHeadModel(ABC):
    """
    Abstract base class for flow-head models.
    """
    def __init__(self, **default_params):
        self.default_params = dict(default_params)

    @abstractmethod
    def head_difference_from_flow(self, rate, **kwargs):
        """
        Compute head difference or one head given the other and flow rate.
        """
        raise NotImplementedError

    @abstractmethod
    def flow_from_heads(self, h_start, h_end, **kwargs):
        """
        Compute flow rate given head difference.
        """
        raise NotImplementedError
    
    def start_head_from_end_and_flow(self, rate, h_end, **kwargs):
        return h_end - self.head_difference_from_flow(rate, **kwargs)

    def end_head_from_start_and_flow(self, rate, h_start, **kwargs):
        return h_start + self.head_difference_from_flow(rate, **kwargs)

    def __call__(self, rate=None, h_start=None, h_end=None, **kwargs):
        if rate is None:
            return self.flow_from_heads(h_start, h_end, **kwargs)
        elif h_start is None:
            return self.start_head_from_end_and_flow(rate, h_end, **kwargs)
        elif h_end is None:
            return self.end_head_from_start_and_flow(rate, h_start, **kwargs)
        else:
            raise ValueError('rate, h_start and h_end cannot be defined all three at the same time')