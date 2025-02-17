
from fluid_functions import single_phase_head_gradient
import numpy as np
from aux_func import inverse_function, fit_parameter
import scipy.constants as spc

# ---------------------------------- #
# setup
# ---------------------------------- #

sep = '\n' + '-' * 20

get_h_from_Q = single_phase_head_gradient
get_Q_from_h = inverse_function(get_h_from_Q, x0=1, bracket=[-1e8, 1e8], vectorize=True)

def test_params():
  eps=0.0005e-3
  D = 124.2e-3
  density = 1000
  viscosity = 0.001
  flow_rate = np.array([360 / spc.hour, 10 / spc.hour])
  L = 100
  inc=0
  K=0
  names = 'eps', 'D', 'density', 'viscosity', 'flow_rate', 'L', 'inc', 'K'
  values = eps, D, density, viscosity, flow_rate, L, inc, K
  return names, values

# ---------------------------------- #
# inverse function test
# ---------------------------------- #

def test_inverse():
  params = dict(zip(*test_params()))

  Q = params.pop('flow_rate')
  dh_true = get_h_from_Q(Q, **params)

  # inverse function
  print('Inverse test:')
  print('Q:', Q)
  print('dh:', dh_true)
  Q2 = get_Q_from_h(dh_true, **params)
  print('Back calculated rate:', Q2)

  assert np.isclose(Q, Q2, rtol=.001).all(), 'Inverse function not working properly'
  return

# ---------------------------------- #
# fit parameters test
# ---------------------------------- #

def test_fit():
  params = dict(zip(*test_params()))

  Q = params.pop('flow_rate')
  dh = get_h_from_Q(Q, **params)

  eps_true = params.pop('eps')
  K_true = params.pop('K')

  # match eps
  print('Match test')

  eps_match = fit_parameter(get_h_from_Q, Q, dh, 'eps', initial_guess=1e-8, 
                            func_kwargs=dict(**params, K=K_true))
  print('Match eps:', eps_match)
  print('True value:', eps_true)

  # match K
  K_match = fit_parameter(get_h_from_Q, Q, dh, 'K', initial_guess=10, 
                            func_kwargs=dict(**params, eps=eps_true))
  print('Match K:', K_match)
  print('True value:', K_true)
  print()

  assert np.isclose(eps_true, eps_match, rtol=1e-3), 'eps match not working'
  assert np.isclose(K_true, K_match, atol=1e-3), 'K match not working'
  return

# ---------------------------------- #
# run tests
# ---------------------------------- #
print(sep)
test_inverse()
print(sep)
test_fit()
print(sep)
print('Test succesful')

