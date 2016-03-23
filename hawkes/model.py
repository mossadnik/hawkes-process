
from scipy import sparse
import numpy as np
from .trainer import Adam
from .simulation import simulateHawkes


class HawkesProcess(object):
    def __init__(self, kernel, mu0, b0,
                 b_pseudo_counts=None,
                 alpha_pi=None, alpha=1e-2):
        '''
        Stochastic EM estimation of marked Hawkes process.

        Args:
          kernel: HawkesKernel for time dependencies
          mu0: array of shape (n_marks, ) with initial background
               Poisson rates for initialization
          b0: matrix of branching ratios (one poisson rate for each
              possible transition) for initialization and prior
          b_pseudo_counts: pseudo counts for prior on branching Poisson rate b
          alpha_pi: array of shape (n_marks + 1, n_marks) specifying
                    the pseudo counts of Dirichlet prior on the prior
                    pi_z on latent variables.
                    The i-th column contains the prior for ancestors of
                    events of type i. Each entry has to be >= 1.
          alpha: learning rate for Adam stochastic gradient optimizer.
        '''
        self.kernel = kernel
        self.n_marks = mu0.size
        self.n_params = self.n_marks * (1 + self.n_marks)
        # global parameters
        self._theta_mu = np.log(mu0)
        self._theta_b = np.log(b0)
        # priors
        self.alpha_b = np.ones_like(b0)
        self.beta_b = np.zeros_like(b0)
        if b_pseudo_counts is not None:
            self.alpha_b += b_pseudo_counts * b0
            self.beta_b = b_pseudo_counts
        z_shape = (self.n_marks + 1, self.n_marks)
        if not alpha_pi:
            self.alpha_pi = np.ones(z_shape)
        else:
        	self.alpha_pi = alpha_pi
        norm = self.alpha_pi.sum(axis=0)[np.newaxis, :]
        self._theta_pi = np.log((self.alpha_pi / norm) / (1. - self.alpha_pi / norm))
        self._trainer = Adam(alpha=alpha)

    @property
    def mu(self):
        return np.exp(self._theta_mu)

    @property
    def b(self):
        return np.exp(self._theta_b)

    @property
    def pi_z(self):
        return 1. / (1. + np.exp(-self._theta_pi))

    def partial_fit(self, t, marks, T=None):
        '''
        Update model parameters using a single sequence as mini-batch.

        Args:
          t: array of event times
          marks: integer array of event types
          T (optional): observation period, has to be > 0 if specified.
                        Inferred as T = t.max() if not specified.
        '''
        T = T or t.max()
        if T == 0:
            raise ValueError("Invalid observation period T=0.\nInferring T requires t.max() > 0.")
        g = self._grad(t, marks, T)
        self._update(self._trainer.update(g))

    def _update(self, delta):
        n = self.n_marks
        pi_size = self._theta_pi.size
        pi_shape = self._theta_pi.shape
        self._theta_mu += delta[:n]
        self._theta_b += delta[n:self.n_params].reshape((n, n))
        self.kernel.update(delta[self.n_params:-pi_size])
        # update theta_pi: project onto constraint surface \sum_i\pi_{ij} = 1
        self._theta_pi += delta[-pi_size:].reshape(pi_shape)
        pi_z = self.pi_z
        pi_z /= pi_z.sum(axis=0)[np.newaxis, :]
        self._theta_pi = np.log(pi_z / (1. - pi_z))

    def _preprocess(self, t, marks):
        '''Compute auxiliary quantities for EM-steps.'''
        # event_mark allows to transform from mark-space to event-space
        # and vice versa
        n = t.size
        data = (np.ones(n), (np.arange(n, dtype=np.int), marks))
        event_mark = sparse.csr_matrix(data, shape=(n, self.n_marks))
        dt = t[np.newaxis, :] - t[:, np.newaxis]
        return dt, event_mark

    def _e_step(self, dt, event_mark, mu, b, pi_z):
        '''Estimate latent event ancestors.'''
        n = dt.shape[0]
        z = np.zeros((n + 1, n))
        z[-1, :] = np.array(event_mark * mu).ravel()  # background rates
        # map b to each pair of events
        _b = np.array(event_mark * b * event_mark.T)
        z[:-1, :] = _b * self.kernel.likelihood(event_mark, dt)
        np.fill_diagonal(z, 0)  # remove spurious self-excitations
        # priors and normalization
        pi_z_event = np.r_[
           np.array(event_mark * pi_z[:-1, :] * event_mark.T),
           np.array(event_mark * pi_z[-1, :]).ravel()[np.newaxis, :]
           ]
        z *= pi_z_event
        z /= z.sum(axis=0)[np.newaxis, :]  # normalize over ancestors
        return z

    def _grad(self, t, marks, T=None):
        '''Compute gradient of likelihood for single sequence.'''
        # init data structures
        dt, event_mark = self._preprocess(t, marks)
        mu, b, pi_z = self.mu, self.b, self.pi_z
        z = self._e_step(dt, event_mark, mu, b, pi_z)
        return self._m_step(z, dt, event_mark, mu, b, pi_z, T)

    def transform(self, t, marks):
        '''
        Transform event sequence to latent ancestor variables.

        Note that results are affected by the prior on z, but not on
        other parameters.

        returns:
          z: (n_marks + 1) x n_marks matrix of ancestor probabilities for
             each event. Each column is a categorical distribution over
             possible ancestors. The last row corresponds to background.
        '''
        dt, event_mark = self._preprocess(t, marks)
        z = self._e_step(dt, event_mark, self.mu, self.b, self.pi_z)
        return z

    def _m_step(self, z, dt, event_mark, mu, b, pi_z, T):
        # background
        grad_mu = (np.array(z[-1, :] * event_mark).ravel() - T * mu) / T
        # number of observed parent and child events
        n_observation = np.array(event_mark.sum(axis=0)).ravel()[:, np.newaxis]
        nu = np.array(event_mark.T * z[:-1, :] * event_mark)  / (n_observation + 1e-8)
        # self-excitation
        grad_b = np.where(n_observation > 0, -(1. + self.beta_b) * b + nu + self.alpha_b - 1., 0)
        grad_kernel = self.kernel.grad(event_mark, z[:-1, :], dt, n_observation, nu)
        # prior on z
        c_pi = np.array(event_mark.T * z[:-1, :] * event_mark)
        c_pi = np.r_[c_pi, np.array(event_mark.T * z[-1, :]).ravel()[np.newaxis, :]]
        grad_pi = (1. - pi_z) * (c_pi + self.alpha_pi - 1.) # add alpha_pi - 1 for conjugate prior
        grad_constraint = pi_z * (1 - pi_z)
        mult = np.dot(grad_pi.ravel(), grad_constraint.ravel()) / np.sum(grad_constraint**2)
        grad_pi -= mult * grad_constraint
        grad_pi *= (n_observation > 0).T
        return np.r_[grad_mu, grad_b.ravel(), grad_kernel, grad_pi.ravel()]

    def simulate(self, T, discrete=False):
        '''
        Simulate event sequence.

        For discrete-time kernels such as NegativeBinomialKernel discrete
        must be set to True.
        '''
        return simulateHawkes(self, T, discrete)
