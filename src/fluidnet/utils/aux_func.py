from scipy.optimize import root_scalar, minimize
import numpy as np

def inverse_function(f, method='brentq', bracket=None, vectorize=True, **root_scalar_kwargs):
    """
    Calculates the inverse of a function f such that f^-1(y) = x.

    Parameters:
    - f: callable
        The function to invert (f(x)).
    - x0: float
        Initial guess for the root-finding method.
    - method: str (default='brentq')
        The root-finding method. Supported methods: 'brentq', 'bisect', etc.
    - bracket: tuple or None
        If using a bracketing method (e.g., 'brentq'), specify (a, b) where f(a) and f(b) must have opposite signs.
    - root_scalar_kwargs (optional): additional parameters to be passed to scipy.optimize.root_scalar

    Returns:
    - x: float
        The value x such that f(x) ≈ y.
    """
    def f_inverse(y, *args,**kwargs):
      def equation(x):
          return f(x, *args,**kwargs) - y
      
      if bracket:
        root_scalar_kwargs['bracket'] = bracket
      result = root_scalar(equation, method=method, **root_scalar_kwargs)
      
      if result.converged:
          return result.root
      else:
          raise ValueError("Root-finding did not converge.")
    
    if vectorize:
      f_inverse = np.vectorize(f_inverse)
    return f_inverse


def fit_parameter(func, x, y, param, initial_guess=0,
                  minimize_kwargs=dict(), func_kwargs=dict()):
  initial_guess = np.asarray(initial_guess)
  def dummy(p):
    diff = (func(x, **{param:p}, **func_kwargs) - y)
    return (diff **2).sum()
  return minimize(dummy, initial_guess, **minimize_kwargs).x
