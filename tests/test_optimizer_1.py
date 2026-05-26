import pytest
import numpy as np
from network import Network
from optimizer import Optimizer
from fluid_functions import single_phase_head_gradient
import pandas as pd

@pytest.fixture
def sample_network():
    edges = [
        ('A', 'B', {'D': 0.1, 'L': 10}),
        ('B', 'C', {'D': 0.1, 'L': 10})
    ]
    node_attributes = {'A': {}, 'B': {}, 'C': {}}
    return Network(edges=edges, node_attributes=node_attributes, potential_from_flow=single_phase_head_gradient)

@pytest.fixture
def sample_optimizer(sample_network):
    edge_params = ['D']
    default_values = [0.1]
    bounds = [(0.05, 0.2)]
    return Optimizer(n=sample_network, edge_params=edge_params, default_values=default_values, bounds=bounds)

def test_propagate_optimization(sample_optimizer):
    optimizer = sample_optimizer
    test_rate = {'A': 1.0, 'C': -1.0} # Simple flow from A to C
    test_head = {'A': 100, 'C': 90}  # Expected heads at nodes
    result = optimizer.optimize(pd.Series(test_rate), pd.Series(test_head))
    
    assert result is not None, "Optimization failed"
    assert result.success, f"Optimization did not converge: {result.message}"

def test_balance_optimization(sample_optimizer):
    optimizer = sample_optimizer
    test_rate = {'A': 1.0}
    test_head = {'C': 90}
    head_bc = {'A': 100,}  # Needed for balance
    
    result = optimizer.optimize(pd.Series(test_rate), pd.Series(test_head), head_bc=head_bc)
    
    assert result is not None, "Optimization failed"
    assert result.success, f"Optimization did not converge: {result.message}"

if __name__ == '__main__':
    sample_network = sample_network()
    sample_optimizer = sample_optimizer(sample_network)
    test_propagate_optimization()
    # test_balance_optimization()