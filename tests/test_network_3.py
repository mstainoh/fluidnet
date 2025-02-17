from network import Network
import numpy as np

if __name__ == '__main__':
   from common import sep, test_network_parameters
else:
   from .common import sep, test_network_parameters

test_network = Network(**test_network_parameters)

#  ---------------------------------- #
# Rate propagation test
# ---------------------------------- #

# from rate test
def test_rate_propagation():

  # From rate test
  n = test_network
  n.debug=False
  rate_bc = {'a': .02, 'b': .03, 'd': -.015}
  node_rates, edge_rates, node_heads = n.propagate_rates(rate_bc=rate_bc, H0=0)
  print('Test propagation:')
  print('\tEdge rates:', edge_rates)
  print('\tNode rates:', node_rates)
  print('\tNode heads:', node_heads)
  print()

  for e in n.edges:
    h1, h2 = map(node_heads.get, e)
    assert h1 >= h2, 'Nodes should have increasing H {}'.format(dict(zip(e, (h1, h2))))

  # reverse network test
  n.reverse_network()
  rate_bc = {k: -v for k, v in rate_bc.items()}
  node_rates, edge_rates, node_heads = n.propagate_rates(rate_bc=rate_bc, from_source=False, H0=0)
  print('Reversed:')
  print('\tEdge rates:', edge_rates)
  print('\tNode rates:', node_rates)
  print('\tNode heads:', node_heads)
  print()
  for e in n.edges:
    h1, h2 = map(node_heads.get, e)
    assert h1 <= h2, 'Nodes should have decreasing H {}'.format(dict(zip(e, (h1, h2))))

# ---------------------------------- #
# Balance calculation
# ---------------------------------- #

def test_balance():
  print('Balance test\n')
  n = test_network
  head_bc = {'a': 100, 'b': 90}
  rate_bc = {'f': -.05}
  n.set_boundary_conditions(head_bc=head_bc, rate_bc=rate_bc)

  Hs = n.balance(tol=1e-14)
  H = dict(zip(n.nodes, Hs))
  
  print('Balanced head:', Hs)

  edge_flows = n.get_edge_flows(H)
  print('Edge flows:', edge_flows)

  node_balance = n.get_node_flows_balance(edge_flows)
  node_balance = dict(zip(n.nodes, node_balance))
  print('rate error:', node_balance)

  non_head_node_balance = [r for i, r in node_balance.items() if i not in head_bc.keys()]
  non_head_node_balance = np.asarray(non_head_node_balance)
  print('Rate balance array (exc. head nodes):', non_head_node_balance)
  print('rate error total (exc. head nodes:)', non_head_node_balance.sum())
  assert np.allclose(non_head_node_balance, 0, atol=1e-7), 'balance error'



# ---------------------------------- #
# Run tests
# ---------------------------------- #
if __name__ == '__main__':
    print(sep)
    test_rate_propagation()

    print(sep)
    test_balance()

    print(sep)
    print('Test succesful')
# ---------------------------------- #
# 
# ---------------------------------- #
