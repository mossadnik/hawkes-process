
from scipy import sparse
import numpy as np


class HawkesProcess(object):
    def __init__(self, mu_bg, b_count, b_mean, mu_count, mu_mean):
        '''
        Class for online estimation of marked Hawkes process.

        Only constant background rates and exponential kernels
        are supported. Online-EM is used with stochastic gradients
        to optimize likelihood.
        '''
        self.n_marks = mu_bg.size
        n = self.n_marks
        n_params = n + n * n + n
        # global parameters
        self._theta = np.zeros(n_params)
        self._theta_mu_bg = self._theta[:n]
        self._theta_b = self._theta[n:-n].reshape((n, n))
        self._theta_mu = self._theta[-n:]
        # gradients
        self._grad = np.zeros(n_params)
        self._grad_mu_bg = self._grad[:n]
        self._grad_b = self._grad[n:-n].reshape((n, n))
        self._grad_mu = self._grad[-n:]
        # priors
        self._alpha_b = b_count * b_mean + 1
        self._beta_b = b_count
        self._alpha_mu = mu_count + 1
        self._beta_mu = mu_count * mu_mean
        # init with priors
        self._theta_mu_bg[:] = np.log(mu_bg)
        self._theta_b[:] = np.log(b_mean)
        self._theta_mu[:] = np.log(mu_mean)

    @property
    def mu_bg(self):
        return np.exp(self._theta_mu_bg)

    @property
    def b(self):
        return np.exp(self._theta_b)

    @property
    def mu(self):
        return np.exp(self._theta_mu)

    def update(self, delta_theta):
        self._theta += delta_theta

    def _preprocess(self, t, marks):
        '''Compute auxiliary quantities for EM-steps.'''
        n_events = t.size
        # event_mark allows to transform from mark-space to event-space
        # and vice versa
        event_mark = sparse.csr_matrix((np.ones(n_events),
                                        (np.arange(n_events, dtype=np.int), marks)),
                                       shape=(n_events, self.n_marks))
        dt = t[np.newaxis, :] - t[:, np.newaxis]
        return dt, event_mark

    def _e_step(self, dt, event_mark, mu_bg, b, mu):
        '''Estimate of latent event ancestors.'''
        z_bg = np.array(event_mark * mu_bg).ravel()  # background rates
        _b = event_mark * b * event_mark.T  # map b to each pair of events
        _mu = (event_mark * mu)[:, np.newaxis]  # map mu to each ancestor
        z = np.where(dt >= 0, _mu * _b * np.exp(-_mu * dt), 0)  # exp decay likelihood
        # remainder is normalization of z, z_bg
        for i in range(z.shape[0]):
            z[i, i] = 0
        norm = z.sum(axis=0) + z_bg  # normalize over ancestors
        z /= norm[np.newaxis, :]
        z_bg /= norm
        return z_bg, z

    def gradient(self, t, marks, T=None):
        '''Compute gradient of likelihood for single observation.'''
        T = T or t.max()
        # init data structures
        dt, event_mark = self._preprocess(t, marks)
        mu_bg, b, mu = self.mu_bg, self.b, self.mu
        z_bg, z = self._e_step(dt, event_mark, mu_bg, b, mu)
        return self._m_step(z, z_bg, dt, event_mark, mu_bg, b, mu, T)

    def transform(self, t, marks):
        '''Transform event sequence to latent ancestor variables.'''
        dt, event_mark = self._preprocess(t, marks)
        z_bg, z = self._e_step(dt, event_mark, self.mu_bg, self.b, self.mu)
        return z_bg, z

    def _m_step(self, z, z_bg, dt, event_mark, mu_bg, b, mu, T):
        # background \partial_\theta L = z\cdot S - T\lambda
        self._grad_mu_bg[:] = (np.array(z_bg * event_mark).ravel() - T * mu_bg) / T
        # summary statistics for likelihood
        nu = np.array(event_mark.T * z * event_mark)
        kappa = np.array(event_mark.T * (z * dt) * event_mark)
        M = np.array(event_mark.sum(axis=0)).ravel()[:, np.newaxis]
        norm = M + 1e-8
        # \partial_{\theta_b}L = -\left[M + \beta_b \right] b + \nu + \alpha_b - 1
        self._grad_b[:] = -(M / norm + self._beta_b) * b + nu /norm + self._alpha_b - 1
        # \partial_{\theta_\mu}L = -\left[\kappa + \beta_\mu\right]\mu + \nu + \alpha_\mu - 1
        norm = norm.sum(axis=1)
        nu = nu.sum(axis=1) / norm
        kappa = kappa.sum(axis=1) / norm
        self._grad_mu[:] = -(kappa + self._beta_mu) * mu + nu + self._alpha_mu - 1.
        return self._grad
