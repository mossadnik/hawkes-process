
from scipy import sparse
import numpy as np
from .trainer import Adam
from .simulation import simulateHawkes

class HawkesProcess(object):
    def __init__(self, kernel, mu0, b0, mu_pseudo_counts=None, b_pseudo_counts=None, alpha=1e-2):
        '''
        Stochastic EM estimation of marked Hawkes process.

        Args:
          kernel: HawkesKernel for time dependencies
          mu0: array of background rates for initialization
          b0: matrix of branching ratios for initialization and prior
          b_pseudo_counts: pseudo counts for prior on b
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
        # gradient
        self._trainer = Adam(alpha=alpha)
        #TODO prior on background rates

    @property
    def mu(self):
        return np.exp(self._theta_mu)

    @property
    def b(self):
        return np.exp(self._theta_b)

    def partial_fit(self, t, marks, T=None):
        '''
        Update model parameters using a single sequence as mini-batch.

        Args:
          t: array of event times
          marks: integer array of event types
          T (float or int > 0): observation period. Inferred as T = t.max() if not specified.
        '''
        T = T or t.max()
        if T == 0:
            raise ValueError("Invalid observation period T=0.\nNote that inference of T works only for sequences of finite temporal extent (t.max() > 0).")
        g = self._grad(t, marks, T)
        self._update(self._trainer.update(g))

    def _update(self, delta):
        n = self.n_marks
        self._theta_mu += delta[:n]
        self._theta_b += delta[n:self.n_params].reshape((n, n))
        self.kernel.update(delta[self.n_params:])

    def _preprocess(self, t, marks):
        '''Compute auxiliary quantities for EM-steps.'''
        # event_mark allows to transform from mark-space to event-space
        # and vice versa
        n = t.size
        data = (np.ones(n), (np.arange(n, dtype=np.int), marks))
        event_mark = sparse.csr_matrix(data, shape=(n, self.n_marks))
        dt = t[np.newaxis, :] - t[:, np.newaxis]
        return dt, event_mark

    def _e_step(self, dt, event_mark, mu, b):
        '''Estimate of latent event ancestors.'''
        z_bg = np.array(event_mark * mu).ravel()  # background rates
        # map b to each pair of events
        _b = np.array(event_mark * b * event_mark.T)
        z = _b * self.kernel.likelihood(event_mark, dt)  # exp decay likelihood
        # remainder is normalization of z, z_bg
        for i in range(z.shape[0]):
            z[i, i] = 0
        norm = z.sum(axis=0) + z_bg  # normalize over ancestors
        z /= norm[np.newaxis, :]
        z_bg /= norm
        return z_bg, z

    def _grad(self, t, marks, T=None):
        '''Compute gradient of likelihood for single observation.'''
        # init data structures
        dt, event_mark = self._preprocess(t, marks)
        mu, b = self.mu, self.b
        z_bg, z = self._e_step(dt, event_mark, mu, b)
        return self._m_step(z, z_bg, dt, event_mark, mu, b, T)

    def transform(self, t, marks):
        '''Transform event sequence to latent ancestor variables.'''
        dt, event_mark = self._preprocess(t, marks)
        z_bg, z = self._e_step(dt, event_mark, self.mu, self.b)
        return z_bg, z

    def _m_step(self, z, z_bg, dt, event_mark, mu, b, T):
        # background
        grad_mu = (np.array(z_bg * event_mark).ravel() - T * mu) / T
        # number of observed parent/ child events
        n_observation = np.array(event_mark.sum(axis=0)).ravel()[:, np.newaxis]
        nu = np.array(event_mark.T * z * event_mark) / (n_observation + 1e-8)
        # self-excitation
        grad_b = np.where(n_observation > 0, -(1. + self.beta_b) * b + nu + self.alpha_b - 1., 0)
        grad_kernel = self.kernel.grad(event_mark, z, dt, n_observation, nu)
        return np.r_[grad_mu, grad_b.ravel(), grad_kernel]

    def simulate(self, T, discrete=False):
        return simulateHawkes(self, T, discrete)
