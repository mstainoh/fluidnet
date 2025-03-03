"""
verify that rate propagation works (both backwards and forward)

verify that balance_solve works and that it gives results that are equal (or very close to) propagate_rates

verify vector compatibility for rate propagation 
"""
from network import Network
import numpy as np

if __name__ == '__main__':
   from common import sep, test_network_parameters
else:
   from .common import sep, test_network_parameters

rtol = 1e-10
def test_network():
  return Network(**test_network_parameters)

#  ---------------------------------- #
# Rate propagation test
# ---------------------------------- #

# from rate test
def test_rate_propagation():

  # From rate test
  n = test_network()
  n.debug=False
  rate_bc = {'a': .02, 'b': .03, 'd': -.015}
  edge_rates, node_heads = n.propagate_rates(rate_bc=rate_bc, H0=0)

  print('Test propagation:')
  print('\tEdge rates:', edge_rates)
  print('\tNode heads:', node_heads)
  print()

  for e in n.edges:
    h1, h2 = map(node_heads.get, e)
    d = n.get_edge_parameters(*e)
    dz = d['inc'] * d['L']
    assert h1 >= h2 + dz, 'Nodes should have increasing H {}'.format(dict(zip(e, (h1, h2))))

  # reverse network test
  n.reverse_network()
  rate_bc = {k: -v for k, v in rate_bc.items()}
  edge_rates, node_heads = n.propagate_rates(rate_bc=rate_bc, from_source=False, H0=0)

  print('Reversed:')
  print('\tEdge rates:', edge_rates)
  print('\tNode heads:', node_heads)
  print()
  for e in n.edges:
    h1, h2 = map(node_heads.get, e)
    d = n.get_edge_parameters(*e)
    dz = d['inc'] * d['L']
    assert h1 <= h2 + dz, 'Nodes should have decreasing H {}'.format(dict(zip(e, (h1, h2))))
  return edge_rates, node_heads

# ---------------------------------- #
# Balance calculation
# ---------------------------------- #

def test_balance(debug=True):
  print('Balance test\n')
  n = test_network()
  n.debug = debug
  head_bc = {'a': 100, 'b': 90}
  rate_bc = {'f': -.05}
  n.set_boundary_conditions(head_bc=head_bc, rate_bc=rate_bc)

  edge_rates, node_heads = n.balance_solve(root_kwargs=dict(tol=1e-10))
  Hs = node_heads

  print()
  print('Balanced head:', Hs)
  print('Edge flows:', edge_rates)

  # rate error
  edge_rates_arr = n.get_edge_flows(Hs)
  node_balance = n.get_node_flows_balance(edge_rates_arr)
  node_balance = dict(zip(n.nodes, node_balance))
  print('rate error:', node_balance)

  # balance error
  non_head_node_balance = [r for i, r in node_balance.items() if i not in head_bc.keys()]
  non_head_node_balance = np.asarray(non_head_node_balance)
  print()
  print('Rate balance array (exc. head nodes):', non_head_node_balance)
  print('rate error total (exc. head nodes:)', non_head_node_balance.sum())
  assert np.allclose(non_head_node_balance, 0, atol=1e-7), 'balance error'

  # comparison vs propagate
  print()
  print('Running comparison between propagate and balance, rtol = {:.2e}'.format(rtol), end=' - ')
  H0 = node_heads['f']
  rate_bc = dict()
  source_nodes = n.get_source_nodes()
  # print(n.is_single_inflow(), n.is_single_outflow())
  for e, rate in edge_rates.items():
     if e[0] in source_nodes:
        rate_bc[e[0]] = rate 

  n.debug = False
  edge_rates_2, node_heads_2 = n.propagate_rates(rate_bc, H0 = H0)
  for e in n.edges:
     assert np.isclose(edge_rates_2[e], edge_rates[e],rtol=rtol), 'Rate discrepancy between propagate and balance for edge {}'.format(e)
  for node in n.nodes:
     assert np.isclose(node_heads[node], node_heads_2[node], rtol=rtol), 'Head discrepancy between propagate and balance for node {}'.format(node)
  print('success')

  return edge_rates, node_heads



def test_vector(debug=False):

  print('Vector test\n')
  n = test_network()
  n.debug=debug
  rate_bc = {'a': np.array([.1, .1]), 'b': np.array([.2, .3])}
  H0 = np.array([0,1])
  edge_rates_2, node_heads_2 =n.propagate_rates(rate_bc=rate_bc, H0=H0)
  print('Rates:', edge_rates_2)
  print('HEads:', node_heads_2)

# ---------------------------------- #
# Run tests
# ---------------------------------- #
if __name__ == '__main__':
    print(sep)
    _ = test_rate_propagation()

    print(sep)
    n_ = test_balance()

    print(sep)
    test_vector()

    print(sep)
    print('Test succesful')
# ---------------------------------- #
# 
# ---------------------------------- #
