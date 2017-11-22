from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from edward.models.random_variable import RandomVariable
from tensorflow.contrib.distributions import Distribution

try:
  from tensorflow.contrib.distributions import NOT_REPARAMETERIZED
except Exception as e:
  raise ImportError("{0}. Your TensorFlow version is not supported.".format(e))


class distributions_StochasticDifferentialEquation(Distribution):
  r"""Stochastic differential equation, based on the Euler-Maruyama (EM)
  discretization, which, for the SDE $dX_{t+1} = dt f(X_t) + dW_t g(X_t)$
  translates to $X_{t+1} ~ N(X_t + dt f(X_t), sqrt(dt) g(X_t))$, where $f$
  and $g$ describe the deterministic flow and stochastic diffusion components
  of the system.

  There is one parameter $dt$ always present, which determines the time step,
  while additional parameters depend on the functions $f$ and $g$.

  #### Examples

  ```python
  # linear SDE
  def fg(x, a, b):
    return a*x, b
  pars = -1.0 , 2.0
  sde = StochasticDifferentialEquation(fd, (-1.0, 2.0), dt=0.2)
  ```
  """
  def __init__(self,
               sde_fun,
               sde_par,
               dt,
               base,
               validate_args=False,
               allow_nan_stats=True,
               name="StochasticDifferentialEquation"):
    """Initialize a batch of stochastic differential equations.

    Args:
      sde_fun: function.
        Function which computes flow and diffusion components of the SDE as a
        function of the state tensor and additional parameters in `sde_par`.
      sde_par: tuple.
        Additional parameters to `sde_fun` required for computing the flow
        and diffusion components of the SDE.
      dt: tf.Tensor.
        Time step, should be positive. Its shape
        determines the number of independent DPs (batch shape).
    """
    self._sde_fun = sde_fun
    self._sde_par = sde_par
    parameters = locals()
    with tf.name_scope(name, values=[dt]):
      with tf.control_dependencies([
          tf.assert_positive(dt),
      ] if validate_args else []):
        self._dt = dt

    super(distributions_StochasticDifferentialEquation, self).__init__(
        dtype=tf.float32,
        reparameterization_type=NOT_REPARAMETERIZED,
        validate_args=validate_args,
        allow_nan_stats=allow_nan_stats,
        parameters=parameters,
        graph_parents=[self._dt, ],
        name=name)

  @property
  def dt(self):
    """Concentration parameter."""
    return self._dt

  def _batch_shape_tensor(self):
    return tf.shape(self.dt)

  def _batch_shape(self):
    return self.dt.shape

  def _event_shape_tensor(self):
    raise NotImplementedError

  def _event_shape(self):
    raise NotImplementedError

  def _sample_n(self, n, seed=None):
    raise NotImplementedError

  def _log_prob(self, x):
    xt = x[:-1]
    f, g = self.sde_fun(xt, *self.sde_pars)
    mu = xt + self._dt * f
    sd = tf.sqrt(self._dt) * g
    return tf.reduce_sum(Normal(loc=mu, scale=sd).log_prob(x[1:]))


# Generate random variable class similar to autogenerated ones from TensorFlow.
def __init__(self, *args, **kwargs):
  RandomVariable.__init__(self, *args, **kwargs)


_name = 'StochasticDifferentialEquation'
_candidate = distributions_StochasticDifferentialEquation
__init__.__doc__ = _candidate.__init__.__doc__
_globals = globals()
_params = {'__doc__': _candidate.__doc__,
           '__init__': __init__}
_globals[_name] = type(_name, (RandomVariable, _candidate), _params)