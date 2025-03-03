"""
verify basic network properties, and that global and individual flow calculations from an H field are equal
"""
from network import Network, get_Q_from_h, get_h_from_Q
import numpy as np

if __name__ == '__main__':
   from common import sep, test_network_parameters
else:
   from .common import sep, test_network_parameters


def test_network():
  return Network(**test_network_parameters)


# ---------------------------------- #
# # test for basic properties
# ---------------------------------- #
def test_basic_network():
  print('Basic test\n')
  n = test_network()
  print('Nodes:', n.nodes)
  print('Edges:', n.edges)
  print('Single outflow:', n.is_single_outflow())
  print('Single inflow:', n.is_single_inflow())
  assert True, ''

# ---------------------------------- #
# build network and test flows test
# ---------------------------------- #

def test_flows():
  print('Single Flow calculation test\n')

  n = test_network()
  Hvalues = [100, 90, 50, 30, 0, -33]
  H = dict(zip('abcdef', Hvalues))
  print('input heads:', H)

  # calculate rates from H - individual
  manual_rates = dict()
  for e in n.edges():
    dh = H[e[1]] - H[e[0]]
    edge_params = n.get_edge_parameters(*e)
    common_params = n.common_parameters
    flow = get_Q_from_h(-dh, **edge_params, **common_params)
    manual_rates[e] = flow

  # calculate rates from H - global
  edge_rates = n.get_edge_flows(H)

  # node test
  print('Edge rates:', edge_rates)
  print('Manual rates:', manual_rates)

  nri, nro = n.get_node_flows(edge_rates, nodes=list('abcdef'))
  print('node rates:', '\n\tin:', nri, '\n\tout:', nro)

  # assert that both individual rates and global rates are equal!
  assert len(manual_rates) == len(edge_rates), 'edges are missing in rate calculation'
  assert all(np.isclose(edge_rates[i], manual_rates[e], rtol=1e-6) for i, e in enumerate(n.edges)), 'some rate have different value'  

# ---------------------------------- #
# run tests
# ---------------------------------- #

if __name__ == '__main__':
  print(sep)
  test_basic_network()
  print(sep)
  test_flows()
  print(sep)
  print('Test succesful')
