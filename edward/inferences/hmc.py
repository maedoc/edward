from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import tensorflow as tf

from edward.inferences.monte_carlo import MonteCarlo
from edward.models import Normal, RandomVariable, Uniform
from edward.util import copy


class HMC(MonteCarlo):
  """Hamiltonian Monte Carlo, also known as hybrid Monte Carlo
  (Duane et al., 1987; Neal, 2011).
  """
  def __init__(self, latent_vars, data=None, model_wrapper=None):
    """
    Examples
    --------
    >>> z = Normal(mu=0.0, sigma=1.0)
    >>> x = Normal(mu=tf.ones(10) * z, sigma=1.0)
    >>>
    >>> qz = Empirical(tf.Variable(tf.zeros([500])))
    >>> data = {x: np.array([0.0] * 10, dtype=np.float32)}
    >>> inference = ed.HMC({z: qz}, data)
    """
    super(HMC, self).__init__(latent_vars, data, model_wrapper)

  def initialize(self, step_size=0.25, n_steps=2, *args, **kwargs):
    """
    Parameters
    ----------
    step_size : float, optional
      Step size of numerical integrator.
    n_steps : int, optional
      Number of steps of numerical integrator.
    """
    self.step_size = step_size
    self.n_steps = n_steps
    self.scope_iter = 0  # a convenient counter for log joint calculations
    return super(HMC, self).initialize(*args, **kwargs)

  def build_update(self):
    """
    Simulate Hamiltonian dynamics using a numerical integrator.
    Correct for the integrator's discretization error using an
    acceptance ratio.
    """
    old_sample = {z: tf.gather(qz.params, tf.maximum(self.t - 1, 0))
                  for z, qz in six.iteritems(self.latent_vars)}

    # Sample momentum.
    old_r_sample = {}
    for z, qz in six.iteritems(self.latent_vars):
      event_shape = qz.get_event_shape()
      normal = Normal(mu=tf.zeros(event_shape), sigma=tf.ones(event_shape))
      old_r_sample[z] = normal.sample()

    # Simulate Hamiltonian dynamics.
    new_sample = old_sample
    new_r_sample = old_r_sample
    for _ in range(self.n_steps):
      new_sample, new_r_sample = leapfrog(old_sample, old_r_sample,
                                          self.step_size, self.log_joint)

    # Calculate acceptance ratio.
    ratio = tf.reduce_sum([0.5 * tf.square(r)
                           for r in six.itervalues(old_r_sample)])
    ratio -= tf.reduce_sum([0.5 * tf.square(r)
                            for r in six.itervalues(new_r_sample)])
    ratio += self.log_joint(new_sample)
    ratio -= self.log_joint(old_sample)

    # Accept or reject sample.
    u = Uniform().sample()
    accept = tf.log(u) < ratio
    sample_values = tf.cond(accept, lambda: list(six.itervalues(new_sample)),
                            lambda: list(six.itervalues(old_sample)))
    if not isinstance(sample_values, list):
      # ``tf.cond`` returns tf.Tensor if output is a list of size 1.
      sample_values = [sample_values]

    sample = {z: sample_value for z, sample_value in
              zip(six.iterkeys(new_sample), sample_values)}

    # Update Empirical random variables.
    assign_ops = []
    variables = {x.name: x for x in
                 tf.get_default_graph().get_collection(tf.GraphKeys.VARIABLES)}
    for z, qz in six.iteritems(self.latent_vars):
      variable = variables[qz.params.op.inputs[0].op.inputs[0].name]
      assign_ops.append(tf.scatter_update(variable, self.t, sample[z]))

    # Increment n_accept (if accepted).
    assign_ops.append(self.n_accept.assign_add(tf.select(accept, 1, 0)))
    return tf.group(*assign_ops)

  def log_joint(self, z_sample):
    """
    Utility function to calculate model's log joint density,
    log p(x, z), for inputs z (and fixed data x).

    Parameters
    ----------
    z_sample : dict
      Latent variable keys to samples.
    """
    if self.model_wrapper is None:
      self.scope_iter += 1

      log_joint = 0.0
      for z, sample in six.iteritems(z_sample):
        z = copy(z, z_sample, scope='prior' + str(self.scope_iter))
        log_joint += tf.reduce_sum(z.log_prob(sample))

      for x, obs in six.iteritems(self.data):
        if isinstance(x, RandomVariable):
          x_copy = copy(x, z_sample, scope='likelihood' + str(self.scope_iter))
          log_joint += tf.reduce_sum(x_copy.log_prob(obs))
    else:
      x = self.data
      log_joint = self.model_wrapper.log_prob(x, z_sample)

    return log_joint


def leapfrog(z_old, r_old, step_size, log_joint):
  z_new = {}
  r_new = {}

  grad_log_joint = tf.gradients(log_joint(z_old), list(six.itervalues(z_old)))
  for i, key in enumerate(six.iterkeys(z_old)):
    z, r = z_old[key], r_old[key]
    r_new[key] = r + 0.5 * step_size * grad_log_joint[i]
    z_new[key] = z + step_size * r_new[key]

  grad_log_joint = tf.gradients(log_joint(z_new), list(six.itervalues(z_new)))
  for i, key in enumerate(six.iterkeys(z_old)):
    r_new[key] += 0.5 * step_size * grad_log_joint[i]

  return z_new, r_new
