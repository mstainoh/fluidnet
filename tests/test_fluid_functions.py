
from fluidnet.utils.fluid_functions import reynolds, find_friction_factor, single_phase_pressure_gradient
import numpy as np
import scipy.constants as spc

if __name__ == '__main__':
   from common import *
else:
   from .common import *

# ---------------------------------- #
# setup
# ---------------------------------- #

K, L, inc = 0, 100, 0
flow_rate = np.array([360 / spc.hour, 10 / spc.hour])

test_params = dict(zip(
  ('eps', 'D', 'density', 'viscosity', 'flow_rate', 'L', 'inc', 'K'),
  (eps, D, density, viscosity, flow_rate, L, inc, K)
))


# ---------------------------------- #
# fluid function tests
# ---------------------------------- #

def test1():
  params = dict(test_params)

  flow_rate = params.pop('flow_rate') # remove flow rate
  
  # get other params for test
  D = params['D']
  density = params['density']
  viscosity = params['viscosity']
  eps = params['eps']

  A = D * D * np.pi / 4
  v = flow_rate / A

  print('Rate:', flow_rate)
  print('Velocity:', v)
  re = reynolds(v, D, density, viscosity)

  # reynolds, friction and head
  print('Reynolds:', re)
  f = find_friction_factor(re, eD=eps/D, fanning=False)
  print('Friction factor:', f)
  print('Checalc value:', 0.0117)
  dP = single_phase_pressure_gradient(flow_rate, **params)
  print('Pressure gradient bar:', dP/spc.bar)
  dh1 = single_phase_pressure_gradient(flow_rate, **params, as_head=True)
  print('Head gradient:', dh1)
  dh2 = single_phase_pressure_gradient(flow_rate, **params, as_head=True)
  assert np.all(dh1 == dh2), 'dhs do not match'

  assert np.allclose(dh1, dP / spc.g / density) , 'dh should be dP / g.rho, dh= {} vs dP/rho/g = {}'.format(dh1, dP / density / spc.g)
  return

# ---------------------------------- #
# run test
# ---------------------------------- #

print(sep)
test1()
print(sep)
print('Test succesful')
