import numpy as np
import networkx as nx
import pandas as pd
import scipy.constants as spc
from scipy.optimize import minimize, least_squares
import warnings
from typing import List, Dict, Optional, Tuple, Union
from network import Network
import itertools

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
        Bounds for each parameter. Should be array of shape (parameters, edges, 2) if edges is provided, or (parameters, 2) otherwise
    edges : Optional[List[Tuple[int, int]]], optional
        List of edges to optimize. Defaults to all edges in the network.
    H0 : float, optional
        Initial head for propagation methods. Defaults to atmospheric pressure.
    """
    
    minimize_kwargs: Dict[str, Union[str, Dict[str, Union[int, bool]]]] = dict(
        method='L-BFGS-B', 
        options={'maxiter': 100, 'disp': True}
    )
    debug: bool = False
 
    def __init__(self, n: Network, edge_params: List[str],
                bounds: np.ndarray,
                edges: Optional[List[Tuple[int, int]]] = None,
                default_values: Optional[List[float]] = None,
                #use_balance: bool = False,
                H0: float = 0):
        
        self.n: Network = n
        self.H0: float = H0
        self.edge_params = edge_params
        lk = len(edge_params)

        bounds = np.asarray(bounds)
        if edges is not None:
            self.edges = edges
            le = len(edges)

            # basic checks
            assert len(edges) > 0, "Edges must be provided"
            assert all(e in n.edges for e in edges), "Edges must be in the network"

            # check bound shape
            assert bounds.shape == (lk, le, 2), "If edges provided, bounds must be provided and match shape (edge_params, edges, 2) = {},{},2".format(lk, le)

            # check default values shape
            if default_values is not None:
                assert np.shape(default_values) == (lk, le), "If edges provided, default values should be shape (edge_params, edges) = ({}, {})".format(lk, le)
        else:
            self.edges = n.edges
            # check bounds shape and resize
            assert bounds.shape == (lk, 2), "If edges not provided, bounds must be provided and match shape (edge_params, 2) = ({}, 2)".format(lk)
            bounds = np.repeat(bounds, len(self.edges), axis=0)
            
            # check default values shape and resize
            if default_values is not None:
                assert np.shape(default_values) == (lk,), "If edges not provided, default values should be shape (edge_params,) = ({})".format(lk)
                default_values = np.repeat(default_values, len(self.edges), axis=0)

        # create indices for df_bounds and default value: parameter -> from -> to -> values
        i1, i2 = map(list, zip(*itertools.product(self.edge_params, self.edges)))
        i2a, i2b = zip(*i2)
        index = pd.MultiIndex.from_tuples(list(zip(i1, i2a, i2b)), names=['param', 'from', 'to'])

        # bounds
        self.bounds = pd.DataFrame(bounds.reshape(-1, 2), index=index, columns=['lb', 'ub'])

        if default_values is None:
            default_values = np.array([[n.edges[e][k] for e in self.edges] for k in self.edge_params], dtype=float).ravel()
        self.default_values = pd.Series(default_values, index=self.bounds.index)

    @staticmethod
    def cost_func(X1: np.ndarray, X2: Union[float, np.ndarray] = 0) -> float:
        """Calculates the cost between two sets of values. Defaults to MAE"""
        er = np.nan_to_num(np.abs(X1 - X2).sum(), 0) / np.count_nonzero(~np.isnan(X1))
        while len(er.shape):
            er = er.sum()
        return er

#%%  edge parameter manipulation
    def reset_parameters(self) -> None:
        """Resets all edge parameters to their default values."""
        for edge_param in self.edge_params:
            self.set_edge_params(edge_param, self.default_values.loc[edge_param].values)

    def set_edge_params(self, edge_param: str, values: Union[float, np.ndarray]) -> None:
        """Assigns new values to specified edge parameters."""

        edges = self.edges
        if np.isscalar(values):
            values = np.repeat(values, len(edges))
        elif len(values) != len(edges):
            raise ValueError("Values array length must match the number of edges")
        else:
            pass
        
        attrs = {e: {edge_param: v} for e, v in zip(edges, values)}
        nx.set_edge_attributes(self.n.G, attrs)  
    
    def set_X(self, X: np.ndarray) -> None:
        """Unfolds X to set edge parameter values"""
        X = np.asarray(X).reshape(len(self.edge_params), -1)
        for arr, edge_param in zip(X, self.edge_params):
            self.set_edge_params(edge_param, arr)

class PropagationOptimizer(Optimizer):
#%%  edge parameter manipulation
    def get_heads(self, test_rate: dict, H0=None) -> Dict[int, float]:
        """
        computes the node heads.
        """
        if H0 is None:
            H0 = self.H0
        _, node_heads = self.n.propagate_rates(test_rate, H0=self.H0)
        
        return node_heads

    def get_error(self, X, test_rates: dict, test_heads: pd.core.frame.DataFrame=None, H0=None, output_frame=False):
        """
        Calculates the cost function.
        Inputs:
        X: values corresponding to params (in order)
        test_rate: DataFrame
        test_heads: DataFrame
        """
        if X is not None:
            self.set_X(X)

        calc_heads = pd.DataFrame(self.get_heads(test_rates))
        calc_heads = calc_heads[test_heads.columns]
        if test_heads is None:
            test_heads = calc_heads.copy() * 0
        errors = calc_heads - test_heads

        # return (dataframe or scalar)
        if output_frame:
            return calc_heads - test_heads

        return self.cost_func(calc_heads, test_heads)
    
    def optimize(self, test_rates: pd.core.frame.DataFrame, test_heads: pd.core.frame.DataFrame) -> Optional[dict]:
        """Runs the optimization process to fit edge parameters."""
        X0: np.ndarray = self.default_values.values
        bounds = self.bounds.values
        test_rates = {k: test_rates[k].values for k in test_rates.columns}
        
        try:
            result = minimize(
                self.get_error, X0, args=(test_rates, test_heads), 
                bounds=bounds, **self.minimize_kwargs
            )
            if self.debug:
                print(result)
        except ValueError as e:
            warnings.warn(f"Optimization failed: {e}")
            return None
        
        return result
