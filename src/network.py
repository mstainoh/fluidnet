"""
Fluid Flow Network Module

This module provides functionality to model and analyze fluid flow networks using
directed graphs. It includes functions to calculate flow and head within the network
and a `Network` class to represent the network structure and perform calculations.

Functions
---------
get_h_from_Q(flow_rate, D, density=1000, viscosity=1e-3, inc=0, eps=0.15e-3, compressibility=0, L=1, K=0, output_array=False, as_head=False)
    Calculate the head from the flow rate using the single-phase head gradient function.
get_Q_from_h(head, D, density=1000, viscosity=1e-3, inc=0, eps=0.15e-3, compressibility=0, L=1, K=0, output_array=False, as_head=False)
    Calculate the flow rate from the head using the inverse of the single-phase head gradient function.

Classes
-------
Network
    Represents a fluid flow network with nodes and edges, allowing calculations of flows and heads.

Imports
-------
from aux_func import inverse_function
from fluid_functions import single_phase_head_gradient
import networkx as nx
import numpy as np
import warnings
from scipy.optimize import root

__all__ = ['Network', 'get_h_from_Q', 'get_Q_from_h']
"""

import pickle
from aux_func import inverse_function
from fluid_functions import single_phase_head_gradient
import networkx as nx
import numpy as np
import warnings
from scipy.optimize import root

__all__ = ['Network', 'get_h_from_Q', 'get_Q_from_h']

# Define flow and head functions
get_h_from_Q = single_phase_head_gradient
get_Q_from_h = inverse_function(
    get_h_from_Q, x0=0, bracket=[-1e6, 1e6], vectorize=False
)

class Network:
    """
    Represents a fluid flow network with nodes and edges.

    This class allows calculations of flows and heads within a directed graph
    where edges represent pipes or connections with associated parameters
    (e.g., resistance, length, etc.), and nodes represent junctions or points
    in the network.

    Parameters
    ----------
    edges : list of tuples
        Each tuple is (node1, node2, edge_data), where edge_data is a dictionary
        of parameters (e.g., length, diameter).
    node_attributes : dict, optional
        A dictionary of node attributes, with node IDs as keys and attribute
        dictionaries as values.
    flow_from_potential : callable, optional
        Function to calculate flow from head difference (default: `get_Q_from_h`).
    potential_from_flow : callable, optional
        Function to calculate head difference from flow (default: `get_h_from_Q`).
    debug : bool, optional
        If True, enables debug mode for verbose outputs (default: False).
    common_parameters : dict, optional
        Common parameters shared across all edges (e.g., fluid density, viscosity).
    
    Note: if a parameter is both in the edge and in the common_parameters, the former will override the latter

    Attributes
    ----------
    G : networkx.DiGraph
        Directed graph representation of the network.
    boundary_conditions : dict
        Stores head and flow boundary conditions for nodes.
    debug : bool
        Debug mode state.
    common_parameters : dict
        Common parameters shared across edges.
    """

    def __init__(
        self, edges, node_attributes=dict(), 
        flow_from_potential=get_Q_from_h, 
        potential_from_flow=get_h_from_Q,
        debug=False, common_parameters=dict()
    ):
        self.common_parameters = dict(common_parameters)
        self.get_flow_from_potential = flow_from_potential
        self.get_potential_from_flow = potential_from_flow

        # Create the graph
        G = nx.DiGraph()
        for e in edges:
            i, j, edge_data = e
            G.add_edge(i, j, **edge_data)
        for node, node_data in node_attributes.items():
            G.nodes[node].update(node_data)
        self.G = G

        self.boundary_conditions = dict()
        self.debug = debug

    def save(self, filename):
        """
        Saves the graph

        Parameters:
        filename (str or Path): The path to the file where the object will be saved.
        """
        with open(filename, 'wb') as file:
            pickle.dump(self.G, file)
    
    @staticmethod
    def load(filename, **kwargs):
        """
        Loads the Graph and creates a network from it.

        Parameters:
        filename (str or Path): The path to the file from which the object will be loaded.
        **kwargs: additional initialization parameters

        Returns:
        Network: The loaded Network object.
        """
        with open(filename, 'rb') as file:
            G = pickle.load(file)
        n = Network([], **kwargs) 
        n.G = G
        return n


    def reverse_network(self):
        """
        Reverse the direction of the entire network.
        """
        self.G = self.G.reverse()

    def __len__(self):
        """
        Return the number of nodes in the network.
        """
        return len(self.G)

    @property
    def nodes(self):
        """
        Get all nodes in the network.
        """
        return self.G.nodes()

    @property
    def edges(self):
        """
        Get all edges in the network.
        """
        return self.G.edges()

    def get_edge_parameters(self, n1, n2):
        """
        Retrieve parameters for a specific edge.

        Parameters
        ----------
        n1 : node
            Starting node of the edge.
        n2 : node
            Ending node of the edge.

        Returns
        -------
        dict
            Edge parameters.
        """
        return self.G.get_edge_data(n1, n2)

    def get_source_nodes(self):
        """
        Get nodes with no incoming edges.

        Returns
        -------
        list
            Source nodes.
        """
        return [n for n, d in self.G.in_degree() if d == 0]

    def get_sink_nodes(self):
        """
        Get nodes with no outgoing edges.

        Returns
        -------
        list
            Sink nodes.
        """
        return [n for n, d in self.G.out_degree() if d == 0]

    def get_middle_nodes(self):
        """
        Get nodes that are neither sources nor sinks.

        Returns
        -------
        list
            Middle nodes.
        """
        sources = set(self.get_source_nodes())
        sinks = set(self.get_sink_nodes())
        return list(set(self.nodes).difference(sources).difference(sinks))

    def is_single_outflow(self):
        """
        Check if all nodes have at most one outgoing edge.

        Returns
        -------
        bool
            True if all nodes have at most one outflow, False otherwise.
        """
        return (np.fromiter((self.G.out_degree(i) for i in self.G.nodes()), int) <= 1).all()

    def is_single_inflow(self):
        """
        Check if all nodes have at most one incoming edge.

        Returns
        -------
        bool
            True if all nodes have at most one inflow, False otherwise.
        """
        return (np.fromiter((self.G.in_degree(i) for i in self.G.nodes()), int) <= 1).all()

    def get_common_parameters(self):
        """
        Get common parameters for all edges.

        Returns
        -------
        dict
            Common parameters (copy).
        
        Note: returns a copy. Use self.common_parameters to get the reference to the actual dictionary.
        """
        return dict(self.common_parameters)

    def get_edge_flow(self, n1, n2, dh, **kwargs):
        """
        Calculate the flow for a given edge and head difference.

        Parameters
        ----------
        n1 : node
            Starting node of the edge.
        n2 : node
            Ending node of the edge.
        dh : float
            Head difference.
        **kwargs: dict(optional)
            additional parameters to be passed to self.get_flow_from_potential.
            Note, these will overwrite edge and common parameters, if duplicated.

        Returns
        -------
        float
            Calculated flow.
        """
        func_kwargs = dict()
        func_kwargs.update(self.common_parameters)
        func_kwargs.update(self.get_edge_parameters(n1, n2))
        func_kwargs.update(kwargs)
        return self.get_flow_from_potential(dh, **func_kwargs)

    def get_edge_dh(self, n1, n2, flow, **kwargs):
        """
        Calculate the head difference for a given edge and flow.

        Parameters
        ----------
        n1 : node
            Starting node of the edge.
        n2 : node
            Ending node of the edge.
        flow : float
            Flow through the edge.
        **kwargs: dict(optional)
            additional parameters to be passed to self.get_potential_from_flow
            Note, these will overwrite edge and common parameters, if duplicated.


        Returns
        -------
        float
            Calculated head difference.
        """
        func_kwargs = dict()
        func_kwargs.update(self.common_parameters)
        func_kwargs.update(self.get_edge_parameters(n1, n2))
        func_kwargs.update(kwargs)
        return self.get_potential_from_flow(flow, **func_kwargs)

    def get_node_flows(self, edge_flows, nodes=None):
        """
        Calculate net inflow and outflow at each node based on edge flows.

        This method determines the flow into and out of each node using the provided
        edge flows. If specific nodes are specified, it returns the flows only for
        those nodes. Does not guarantee node balance.

        Parameters
        ----------
        edge_flows : array-like
            Flow rates for all edges in the network. The order must match the order
            of edges in `self.edges`.
        nodes : list or array-like, optional
            List of specific nodes for which to return flow values. If not provided,
            flows for all nodes are returned.

        Returns
        -------
        flow_in : np.ndarray
            Array of inflow rates for each node.
        flow_out : np.ndarray
            Array of outflow rates for each node.

        Notes
        -----
        - Edge flows must match the order of edges in the graph.
        - The method uses the adjacency matrix of the graph to determine node connectivity.
        """

        edge_flows = np.asarray(edge_flows).flatten()
        nfrom, nto = nx.adjacency_matrix(self.G).nonzero()
        flow_in = np.zeros(len(self), dtype=float)
        flow_out = flow_in.copy()
        for n1, n2, flow in zip(nfrom, nto, edge_flows):
          flow_in[n2] += flow
          flow_out[n1] -= flow

        if nodes is None:
          return flow_in, flow_out
        else:
          all_nodes = list(self.nodes())
          node_order = np.fromiter(map(all_nodes.index, np.atleast_1d(nodes)), int)
          return flow_in[node_order], flow_out[node_order]

    def get_edge_flows(self, Hs):
        """
        Calculate flows for all edges given head differences at nodes.

        Parameters
        ----------
        Hs : dict
            Node head values.

        Returns
        -------
        np.ndarray
            Array of edge flows.
        """
        def sub(e):
            n1, n2 = e
            dh = Hs[n2] - Hs[n1]
            return self.get_edge_flow(n1, n2, dh)

        return np.fromiter(map(sub, self.edges), float)

    def get_node_flows_balance(self, edge_flows):
        """
        Calculate the net flow balance at each node.

        Parameters
        ----------
        edge_flows : array-like
            Flow rates through edges.

        Returns
        -------
        np.ndarray
            Net flow balance for each node.
        """
        flow_in, flow_out = self.get_node_flows(edge_flows)
        rate_bc = self.boundary_conditions.get('rate', dict())
        flow_bc = np.fromiter((rate_bc.get(n, 0) for n in self.nodes()), float)
        return flow_in + flow_out + flow_bc

    def set_boundary_conditions(self, head_bc=None, rate_bc=None, mix_bc=None, check=False):
        """
        Set boundary conditions for the network.

        Parameters
        ----------
        head_bc : dict, optional
            Boundary conditions for head (node: value).
        rate_bc : dict, optional
            Boundary conditions for flow rates (node: value).
        mix_bc : dict, optional
            Mixed boundary conditions (not implemented).
        check : bool, optional
            If True, validate the consistency of boundary conditions.
        """
        if head_bc:
          self.boundary_conditions['head'] = head_bc
        if rate_bc:
          self.boundary_conditions['rate'] = rate_bc
        if mix_bc:
          raise NotImplementedError('mix BC not implemented yet')
          self.boundary_conditions['mix'] = mix_bc

        if check:
          nodes_with_bc = np.hstack([list(bc.keys()) for bc in self.boundary_conditions.values()])
          nodes_without_bc = list(set(self.nodes()).difference(nodes_with_bc))
          sink_nodes = self.get_sink_nodes()
          source_nodes = self.get_source_nodes()
          middle_nodes = self.get_middle_nodes()
          head_bc_nodes = list(self.boundary_conditions.get('head', dict()).keys())

          # no duplicated BC conditions
          unique_elements, counts = np.unique(nodes_with_bc, return_counts=True)
          ar = unique_elements[counts>1]
          if ar.size:
            warnings.warn(f'There are nodes with duplicated boundary conditions, this could lead to errors.  Nodes: {ar}',)

          # all source nodes have BC
          ar = np.intersect1d(nodes_without_bc, source_nodes)
          if ar.size:
            warnings.warn(f'There are source nodes with no boundary conditions, this could lead to errors. Nodes: {ar}',)

          # all sink nodes have BC
          ar = np.intersect1d(nodes_without_bc, sink_nodes)
          if ar.size:
            warnings.warn(f'There are sink nodes with no boundary conditions, this could lead to errors. Nodes: {ar}',)

          # no middle nodes with head BC
          ar = np.intersect1d(head_bc_nodes, middle_nodes)
          if ar.size:
            warnings.warn(f'There are middle nodes with head boundary conditions, this could lead to errors. Nodes: {ar}',)

          # Rate sign for rate BC is positive in sources, negative in sinks:
          rate_bc = self.boundary_conditions.get('rate', dict())
          positive_sinks = np.intersect1d([k for k, v in rate_bc.items() if v > 0], sink_nodes)
          negative_sources = np.intersect1d([k for k, v in rate_bc.items() if v < 0], source_nodes)
          if negative_sources.size:
            warnings.warn(f'Source nodes should have positive rate:\n {negative_sources}')
          if positive_sinks.size:
            warnings.warn(f'Sink nodes should have negative rate:\n {positive_sinks}')

        return self


    def propagate_rates(self, rate_bc=None, from_source=True, H0=0):
        """
        Propagate flow rates and optionally calculate node heads in the network.

        This method is valid only for graphs with single inflow or outflow per node.
        It propagates flow rates through the network based on boundary conditions 
        and optionally calculates the head (pressure) at each node.

        Parameters
        ----------
        rate_bc : dict, optional
            Boundary conditions for node flow rates (node: rate). If not provided, 
            uses the boundary conditions defined in the object.
        from_source : bool, optional
            If True, propagates rates starting from source nodes (default: True).
            If False, propagates rates starting from sink nodes.
        H0 : float, optional
            Initial head (pressure) at the starting or ending node (default: 0).
            If None, skips head calculation.

        Returns
        -------
        node_rates : dict
            Flow rates at each node (node: rate).
        edge_rates : dict
            Flow rates through each edge (edge: rate).
        node_heads : dict
            Heads (pressure values) at each node (node: head). Returns NaN for 
            nodes where head calculation is skipped.

        Raises
        ------
        ValueError
            If the graph is not a valid single-inflow or single-outflow graph.
        AssertionError
            If required boundary conditions are missing for sources or sinks.
        """
        # Initialize rates and heads
        node_rates = dict().fromkeys(self.nodes, 0)
        if rate_bc is None:
            rate_bc = self.boundary_conditions['rate']
        node_rates.update(rate_bc)

        # Ensure the graph is valid for propagation
        if self.is_single_outflow() and from_source:
            assert set(self.get_source_nodes()).issubset(node_rates.keys()), (
                "Some source nodes do not have a rate boundary condition"
            )
        elif self.is_single_inflow() and not from_source:
            assert set(self.get_sink_nodes()).issubset(node_rates.keys()), (
                "Some sink nodes do not have a rate boundary condition"
            )
        else:
            raise ValueError(
                "Graph must be either single outflow with from_source=True or "
                "single inflow with from_source=False"
            )

        # Initialize edge rates and node heads
        edge_rates = dict().fromkeys(self.edges(), 0)
        node_heads = dict().fromkeys(self.nodes, np.nan)

        # Propagate rates forward
        if from_source:
            for n1 in nx.topological_sort(self.G):
                for edge in self.G.out_edges(n1):
                    n2 = edge[1]
                    node_rates[n2] += node_rates[n1]
                    edge_rates[(n1, n2)] = node_rates[n1]

        # Propagate rates backward
        else:
            for n2 in nx.topological_sort(self.G.reverse()):
                for edge in self.G.in_edges(n2):
                    n1 = edge[0]
                    node_rates[n1] += node_rates[n2]
                    edge_rates[(n1, n2)] = node_rates[n2]

        # Skip head calculation if H0 is None
        if H0 is None:
            return node_rates, edge_rates, node_heads

        # Calculate node heads (backward propagation)
        if from_source:
            last_node = list(nx.topological_sort(self.G))[-1]
            node_heads[last_node] = H0
            for n2 in nx.topological_sort(self.G.reverse()):
                for edge in self.G.in_edges(n2):
                    edge_rate = edge_rates[edge]
                    dh = self.get_edge_dh(*edge, edge_rate)
                    node_heads[edge[0]] = node_heads[edge[1]] - dh
                    if self.debug:
                        print(edge, dh, edge_rate)
                        print(node_heads)

        # Calculate node heads (forward propagation)
        else:
            first_node = list(nx.topological_sort(self.G))[0]
            node_heads[first_node] = H0
            for n1 in nx.topological_sort(self.G):
                for edge in self.G.out_edges(n1):
                    edge_rate = edge_rates[edge]
                    dh = self.get_edge_dh(*edge, edge_rate)
                    node_heads[edge[1]] = node_heads[edge[0]] + dh

        return node_rates, edge_rates, node_heads


    def balance(self, **root_kwargs):
        """
        Balance node heads given boundary conditions and flow constraints.

        Solves for node heads that satisfy flow balance equations while respecting 
        the boundary conditions for head and flow rates.

        Parameters
        ----------
        **root_kwargs : dict
            Additional arguments to pass to `scipy.optimize.root`.

        Returns
        -------
        np.ndarray
            Array of head values at each node.

        Raises
        ------
        AssertionError
            If boundary conditions are missing.
        """
        # Initialize head and rate vectors
        Hs = np.zeros(len(self))

        # Validate boundary conditions
        assert self.boundary_conditions, "The network has no boundary conditions"
        bc_head = self.boundary_conditions.get('head', dict())
        bc_rate = self.boundary_conditions.get('rate', dict())

        # Prepare head and rate masks
        nodes = list(self.nodes())
        nodes_head_mask = np.zeros(len(self), dtype=bool)
        for n, head in bc_head.items():
            ix = nodes.index(n)
            nodes_head_mask[ix] = True
            Hs[ix] = head

        node_rates = np.zeros(len(self), dtype=float)
        for n, rate in bc_rate.items():
            ix = nodes.index(n)
            node_rates[ix] = rate

        # Define the error function
        def dummy_error(x):
            Hs[~nodes_head_mask] = x
            edge_flows = self.get_edge_flows(dict(zip(nodes, Hs)))
            flow_in, flow_out = self.get_node_flows(edge_flows)
            return (flow_in + flow_out + node_rates)[~nodes_head_mask]

        # Solve for non-head node values
        x0 = Hs[~nodes_head_mask]
        Hs_sub = root(dummy_error, x0, **root_kwargs).x
        Hs[~nodes_head_mask] = Hs_sub

        return Hs

    def balance_solve(self, head_bc=None, rate_bc=None, mix_bc=None, root_kwargs=dict(), as_dict=True):
        """
        Solves a system for a given set of boundary conditions.
        
        Equivalent to:
            set_boundary_conditions
            solve for node heads
            get edge rates
            get node rates
            return node_rates, edge_rates, node_heads

        Parameters
        ----------
        head_bc : dict, optional
            Boundary conditions for head (node: value).
        rate_bc : dict, optional
            Boundary conditions for flow rates (node: value).
        mix_bc : dict, optional
            Mixed boundary conditions.
        root_kwargs : dict
            Additional arguments to pass to `scipy.optimize.root`.
        as_dict: bool
            if True, returns the output as dictionary. Otherwise as array. Default is True.

        Returns
        -------
        node_rates : dict or array
            Flow rates at each node (node: rate)
        edge_rates : dict or array
            Flow rates through each edge (edge: rate).
        node_heads : dict or array
            Heads (pressure values) at each node (node: head).
        
        Note: node and edge values are in the same order as self.nodes and self.edges, respectively.

        Raises
        ------
        AssertionError
            If boundary conditions are missing.

        """
        self.set_boundary_conditions(head_bc=head_bc, rate_bc=rate_bc, mix_bc=mix_bc)
        node_heads = self.balance(**root_kwargs)
        edge_rates = self.get_edge_flows(dict(zip(self.nodes, node_heads)))
        _, node_rates = self.get_node_flows(edge_rates)
        if as_dict:
            return dict(zip(self.nodes, node_rates)), dict(zip(self.edges, edge_rates)), dict(zip(self.nodes, node_heads))
        return node_rates, edge_rates, node_heads


    def get_head_from_edge_rates(self, edge_rates, end_head=0):
        """
        Calculate node heads from edge rates.

        Parameters
        ----------
        edge_rates : dict
            Flow rates through each edge (edge: rate).
        end_head : float, optional
            Head value at the last node for backward propagation (default: 0).

        Returns
        -------
        np.ndarray
            Array of node head values.

        Raises
        ------
        AssertionError
            If the graph is not a valid single-outflow graph.
        """
        assert self.is_single_outflow(), "All nodes should have at most one out edge"

        out = np.zeros(len(self)) + end_head
        for i in nx.topological_sort(self.G.reverse()):
            for e in self.G.out_edges(i, data=True):
                j = e[1]
                rate = edge_rates[(i, j)]
                dh = self.get_edge_dh(e, rate)
                out[i] = out[j] + dh

        return out

  # # -----------------------
  # # Advanced calculation methods

  # def propagate_rates(self, rate_bc=None, from_source=True, H0=0):
  #   # valid only for single out/in edge graphs
  #   # In this case, we can propagate source rates (single out) or sink rates (single in)
  #   # return node_rates, edge_rates, node_heads (dictionaries, using the same order as self.G)

  #   # get sources, sinks and boundaries
  #   node_rates = dict().fromkeys(self.nodes, 0)
  #   if rate_bc is None:
  #     rate_bc = self.boundary_conditions['rate']
  #   node_rates.update(rate_bc)

  #   # check that all sources (sinks) have rates and the net is single outflow (inflow)
  #   if self.is_single_outflow() and from_source:
  #     # verify all sources have rates
  #     assert set(self.get_source_nodes()).issubset(rate_bc.keys()), 'some source nodes do not have a rate boundary condition'
  #   elif self.is_single_inflow() and not from_source:
  #     assert set(self.get_sink_nodes()).issubset(rate_bc.keys()), 'some source nodes do not have a rate boundary condition'
  #   else:
  #     raise ValueError(
  #         'Graph must be either single outflow with from_source=True '
  #         'or single inflow with from_source=False')

  #   edge_rates = dict().fromkeys(self.edges(), 0)
  #   # edge_dhs = dict().fromkeys(self.edges(), 0)
  #   node_heads = dict().fromkeys(self.nodes, np.nan)

  #   # propagate rates forward:
  #   if from_source:
  #     for n1 in nx.topological_sort(self.G):
  #       for edge in self.G.out_edges(n1):
  #         n2 = edge[1]
  #         node_rates[n2] += node_rates[n1]
  #         edge_rate = node_rates[n1]
  #         edge_rates[(n1, n2)] = edge_rate
  #         # edge_dhs[(n1, n2)] = self.get_edge_dh(*edge, edge_rate)
    
  #   # propagate rate backwards
  #   else:
  #     for n2 in nx.topological_sort(self.G.reverse()):
  #       for edge in self.G.in_edges(n2):
  #         n1 = edge[0]
  #         node_rates[n1] += node_rates[n2]
  #         edge_rate = node_rates[n2]
  #         edge_rates[(n1, n2)] = edge_rate
  #         # edge_dhs[(n1, n2)] = self.get_edge_dh(*edge, edge_rate)

  #   # head calculation
  #   if H0 is None:
  #     pass # skip head calculation

  #   # propagate heads backwards:
  #   elif from_source:
  #     last_node = list(nx.topological_sort(self.G))[-1]
  #     node_heads[last_node] = H0
  #     for n2 in nx.topological_sort(self.G.reverse()):
  #       for edge in self.G.in_edges(n2):
  #         edge_rate = edge_rates[edge]
  #         dh = self.get_edge_dh(*edge, edge_rate)
  #         node_heads[edge[0]] = node_heads[edge[1]] - dh
  #         if self.debug:
  #           print(edge, dh, edge_rate)
  #           print(node_heads)
    
  #   # or propagate heads forwards:
  #   else:
  #     first_node = list(nx.topological_sort(self.G))[0]
  #     node_heads[first_node] = H0
  #     for n1 in nx.topological_sort(self.G):
  #       for edge in self.G.out_edges(n1):
  #         edge_rate = edge_rates[edge]
  #         dh = self.get_edge_dh(*edge, edge_rate)
  #         node_heads[edge[1]] = node_heads[edge[0]] + dh


  #   return node_rates, edge_rates, node_heads

  # def balance(self, **root_kwargs):
  #   # get Hs from a set of boundary conditions, generic method
  #   # Idea: fix head for nodes with head boundary conditions
  #   # solve balance

  #   # head vector (in node order)
  #   Hs = np.zeros(len(self))

  #   # get boundaries
  #   assert self.boundary_conditions, 'The network has no boundary conditions'
  #   bc_head = self.boundary_conditions.get('head', dict())
  #   bc_rate = self.boundary_conditions.get('rate', dict())
  #   bc_other = self.boundary_conditions.get('mix', dict())

  #   # get indices for head nodes and set constant head values
  #   nodes = list(self.nodes())
  #   nodes_head_mask = np.zeros(len(self), dtype=bool)
  #   for n, head in bc_head.items():
  #     ix = nodes.index(n)
  #     nodes_head_mask[ix] = True
  #     Hs[ix] = head

  #   # get rate vector
  #   node_rates = np.zeros(len(self), dtype=float)
  #   for n, rate in bc_rate.items():
  #     ix = nodes.index(n)
  #     node_rates[ix] = rate

  #   # set guess vector using non-head nodes
  #   x0 = Hs[~nodes_head_mask]

  #   # calculate error adjusting values for non-head boundary nodes
  #   def dummy_error(x):
  #     Hs[~nodes_head_mask] = x
  #     edge_flows = self.get_edge_flows(dict(zip(nodes, Hs)))
  #     flow_in, flow_out = self.get_node_flows(edge_flows)
  #     node_rate_balance = flow_in + flow_out + node_rates
  #     # if self.debug:
  #     #   print('flow_in', flow_in)
  #     #   print('flow_out', flow_out)
  #     #   print('node_rate_balance', node_rate_balance)
  #     #   print('Hs:', Hs)
  #     #   print()


  #     return node_rate_balance[~nodes_head_mask]

  #   Hs_sub = root(dummy_error, x0, **root_kwargs).x
  #   Hs[~nodes_head_mask] = Hs_sub

  #   return Hs


  # def get_head_from_edge_rates(self, edge_rates, end_head=0):
  #   # verify that nodes have a single output
  #   assert self.is_single_outflow(), 'All nodes should have at most one out edge'

  #   out = np.zeros(len(self)) + end_head
  #   for i in nx.topological_sort(self.G.reverse()):
  #     for e in self.G.out_edges(i, data=True):
  #       j = e[1]
  #       rate = edge_rates[(i, j)]
  #       dh = self.get_edge_dh(e, rate, data=True)
  #       out[i] = out[j] + dh

  #   return out