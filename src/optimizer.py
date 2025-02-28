import numpy as np
import networkx as nx
import pandas as pd
import scipy.constants as spc
from scipy.optimize import minimize, least_squares
import warnings
from typing import List, Dict, Optional, Tuple, Union
from network import Network

class Optimizer:
    """
    Optimizer class for fitting network parameters to observed data.
    
    This class adjusts edge parameters in a fluid flow network to minimize the error
    between calculated and observed node heads using optimization techniques.
    
    Parameters
    ----------
    n : Network
        The network object to optimize.
    edge_params : List[str]
        List of edge parameters to optimize.
    default_values : Optional[List[float]], optional
        Default values for the parameters.
    bounds : Optional[List[Tuple[float, float]]], optional
        Bounds for each parameter.
    edges : Optional[List[Tuple[int, int]]], optional
        List of edges to optimize. Defaults to all edges in the network.
    H0 : float, optional
        Initial head for propagation methods. Defaults to atmospheric pressure.
    """
    
    minimize_kwargs: Dict[str, Union[str, Dict[str, Union[int, bool]]]] = dict(
        method='L-BFGS-B', 
        options={'maxiter': 10000, 'disp': True}
    )
    debug: bool = False
 
    def __init__(self, n: Network, edge_params: List[str], default_values: Optional[List[float]] = None,
                 bounds: Optional[List[Tuple[float, float]]] = None,
                 edges: Optional[List[Tuple[int, int]]] = None,
                 use_balance: bool = False,
                 H0: float = 10):
        if bounds is None or len(bounds) != len(edge_params):
            raise ValueError("Bounds must be provided and match the length of edge_params")
        
        if default_values is None:
            default_values = [b[0] for b in bounds]
        if len(default_values) != len(edge_params):
            raise ValueError("Default values must match the length of edge_params")
        
        self.n: Network = n
        self.H0: float = H0
        self.edges: List[Tuple[int, int]] = edges if edges is not None else list(n.edges)
        self.edge_params: List[str] = edge_params
        self.default_values: Dict[str, float] = dict(zip(edge_params, default_values))
        self.bounds: Dict[str, Tuple[float, float]] = dict(zip(edge_params, bounds))
        self.use_balance = use_balance

    @property
    def default_X(self) -> np.ndarray:
        """Returns the default parameter values as a flattened array."""
        return np.hstack([[self.default_values[param]] * len(self.edges) for param in self.edge_params])

    @property
    def current_X(self) -> np.ndarray:
        """Returns the current network parameter values as a flattened array."""
        return np.hstack([[self.n.G.edges[e][param] for e in self.edges] for param in self.edge_params])
    
    @property
    def bound_tuples(self) -> np.ndarray:
        """Returns the bounds formatted for optimization functions."""
        return np.array([b for param in self.edge_params for b in self.bounds[param]]).reshape(-1, 2)

    @staticmethod
    def cost_func(X1: np.ndarray, X2: Union[float, np.ndarray] = 0) -> float:
        """Calculates the cost between two sets of values. Defaults to MAE"""
        return np.abs(X1 - X2).sum() / np.size(X1)

    def reset_parameters(self) -> None:
        """Resets all edge parameters to their default values."""
        for edge_param, value in self.default_values.items():
            self.set_edge_params(edge_param, value, edges=self.edges)

    def set_edge_params(self, edge_param: str, values: Union[float, np.ndarray],
                        edges: Optional[List[Tuple[int, int]]] = None) -> None:
        """Assigns new values to specified edge parameters."""
        if edges is None:
            edges = self.edges
        
        if np.isscalar(values):
            values = np.repeat(values, len(edges))
        elif len(values) != len(edges):
            raise ValueError("Values array length must match the number of edges")
        
        attrs = {e: {edge_param: v} for e, v in zip(edges, values)}
        nx.set_edge_attributes(self.n.G, attrs)  
    
    def set_X(self, X: np.ndarray) -> None:
        """Sets the edge parameters to the values in X."""
        for i, edge_param in enumerate(self.edge_params):
            param_values = X[i * len(self.edges): (i + 1) * len(self.edges)]
            self.set_edge_params(edge_param, param_values)

    def get_heads(self,test_rate: Dict[int, float],
                      head_bc: Optional[Dict[int, float]] = None,
                       ) -> Dict[int, float]:
        """
        computes the node heads.
        """
        if self.use_balance:
            _, node_heads = self.n.balance_solve(rate_bc=test_rate, head_bc=head_bc, as_dict=True)
        else:
            _, node_heads = self.n.propagate_rates(test_rate, H0=self.H0)
        
        return node_heads

    def get_error(self, X, test_rates, test_heads, head_bc=None, output_frame=False):
        """
        Calculates the cost function.
        Inputs:
        X: values corresponding to params (in order) - see below
        test_rate: list of rate dictionary (node: array)
        test_pressure: list of pressure_dictionary (node: array)
        """
        self.set_X(X)
        errors = pd.Series()
        print(test_rates, test_heads)
        for test_rate, test_head in zip(test_rates, test_heads):
            # get propagated rates
            node_heads = self.get_heads(test_rate, head_bc)

            # intersect common nodes
            common_nodes = list(set(test_head.keys()).intersection(node_heads.keys()))

            # calculate error
            error = pd.Series(node_heads).loc[common_nodes] - pd.Series(test_head).loc[common_nodes]
            errors = errors.add(error.abs(), fill_value=0)
            
        # return (dataframe or scalar)
        if output_frame:
            return errors

        return self.cost_func(errors)
    
    def optimize(self, test_rates: List[Dict[int, float]], test_heads: List[Dict[int, float]],
                 head_bc: Optional[Dict[int, float]] = None,
                 bounds: Optional[np.ndarray] = None, use_balance: bool = False) -> Optional[dict]:
        """Runs the optimization process to fit edge parameters."""
        X0: np.ndarray = self.current_X
        bounds = bounds if bounds is not None else self.bound_tuples
        
        try:
            result = minimize(
                self.get_error, X0, args=(test_rates, test_heads, head_bc, use_balance), 
                bounds=bounds, **self.minimize_kwargs
            )
            if self.debug:
                print(result)
        except ValueError as e:
            warnings.warn(f"Optimization failed: {e}")
            return None
        
        return result
