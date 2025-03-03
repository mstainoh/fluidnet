"""
verify set_boundary_conditions method
"""
from network import Network
import warnings

if __name__ == '__main__':
   from common import sep, test_network_parameters
else:
   from .common import sep, test_network_parameters

# ---------------------------------- #
# setup
# ---------------------------------- #

test_network = Network(**test_network_parameters)
# ---------------------------------- #
#  boundary condition tests
# ---------------------------------- #

def test_bc():
  print('BC test\n')
  n = test_network
  
  head_bc = {'a': 100, 'b': 90,}
  bad_rate_bc = {'d': 0.2, 'a': -0.3}
  rate_bc = {'f': -.5}
  
  #Improper bc set
  with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")  # Capture all warnings
        n.set_boundary_conditions(head_bc=head_bc, rate_bc=bad_rate_bc, check=True)
        print('bc 1:', n.boundary_conditions)      
        # Print captured warnings
        print('Warnings:')
        for warning in caught_warnings:
            print(f"\t{warning.message}")
 
  print()
  # Proper bc set
  n.set_boundary_conditions(head_bc=head_bc, rate_bc=rate_bc, check=True)
  print('bc 2:', n.boundary_conditions)
  
  assert n.boundary_conditions['head'] == head_bc, 'head bc not set correctly'
  assert n.boundary_conditions['rate'] == rate_bc, 'rate bc not set correctly'
  #assert n.boundary_conditions['mix'] == {}, 'mix bc not set correctly'
  return

# ---------------------------------- #
# run tests
# ---------------------------------- #
if __name__ == '__main__':
    print(sep)
    test_bc()
    print(sep)
    print('Test succesful')
