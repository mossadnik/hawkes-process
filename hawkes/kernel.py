import numpy as np
from scipy.special import psi, gamma
from scipy import stats


class ExponentialKernel(object):
    def __init__(self, mu0, pseudo_counts=None):
        '''
        Exponential kernel for Hawkes process.

        Args:
          mu0: matrix of decay rates for initialization and prior mean.
          pseudo_counts: matrix of pseudo counts for strength of prior.
        '''
        self._theta = np.log(mu0)
        self.n_params = mu0.size
        self.beta = np.zeros_like(mu0)
        self.alpha = np.ones_like(mu0)
        if pseudo_counts is not None:
            self.beta[:] = mu0 * pseudo_counts
            self.alpha += pseudo_counts

    @property
    def mu(self):
        return np.exp(self._theta)

    @property
    def theta(self):
        return self._theta.ravel()

    def update(self, delta):
        self._theta += delta.reshape(self._theta.shape)

    def likelihood(self, event_mark, dt):
        '''Compute likelihood of observed event sequence.'''
        _mu = event_mark * self.mu * event_mark.T
        return np.where(dt >= 0, _mu * np.exp(-_mu * dt), 0)

    def grad(self, event_mark, z, dt, n_observation, nu):
        '''Compute gradient with respect to theta.'''
        kappa = np.array(event_mark.T * (z * dt) * event_mark) / (n_observation + 1e-8)
        grad = np.where(n_observation > 0, -(kappa + self.beta) * self.mu + nu + self.alpha - 1., 0)
        return grad.ravel()

    def dist(self, i, j):
        mu = self.mu[i, j]
        return stats.expon(scale=1. / mu)


class PoissonKernel(object):
    def __init__(self, mu0):
        self.n_params = mu0.size
        self._theta = np.log(mu0)

    @property
    def mu(self):
        return np.exp(self._theta)

    def likelihood(self, event_mark, dt):
        mu = np.array(event_mark * self.mu * event_mark.T)
        return stats.poisson(mu).pmf(dt.astype(int))

    def grad(self, event_mark, z, dt, n_observation, nu):
        n = n_observation + 1e-8
        kappa = np.array(event_mark.T * (z * dt) * event_mark) / n
        return np.where(n_observation > 0, kappa - nu * self.mu, 0.).ravel()

    def update(self, delta):
        self._theta += delta.reshape(self._theta.shape)

    def dist(self, i, j):
        return stats.poisson(self.mu[i, j])


class NegativeBinomialKernel(object):
    def __init__(self, r0, p0):
        '''Negative Binomial Kernel with mean/var parametrization'''
        self.n_params = 2 * r0.size
        self._theta_p = np.log(p0 / (1. - p0))
        self._theta_r = np.log(r0)

    @property
    def r(self):
        return np.exp(self._theta_r)

    @property
    def p(self):
        return 1. / (1. + np.exp(-self._theta_p))

    @property
    def mu(self):
        p = self.p
        return self.r * p / (1. - p)

    @property
    def sigma(self):
        p = self.p
        return np.sqrt(self.r * p) / (1. - p)

    def update(self, delta):
        n = self._theta_p.size
        shape = self._theta_p.shape
        self._theta_r += delta[:n].reshape(shape)
        self._theta_p += delta[n:].reshape(shape)

    def likelihood(self, event_mark, dt):
        r = np.array(event_mark * self.r * event_mark.T)
        p = np.array(event_mark * self.p * event_mark.T)
        return stats.nbinom(r, 1. - p).pmf(dt.astype(int))

    def grad(self, event_mark, z, dt, n_observation, nu):
        _dt = np.maximum(dt, 0.)
        n = n_observation + 1e-8
        r, p = self.r, self.p
        _psi = np.array(event_mark.T *
                        (z * psi(_dt + np.array(event_mark * self.r * event_mark.T))) * 
                        event_mark) / n
        kappa = np.array(event_mark.T * (z * _dt) * event_mark) / n
        grad_r = np.where(n_observation > 0, r * (_psi + nu * (-psi(r) + np.log(1. - p))), 0.)
        grad_p = np.where(n_observation > 0, -nu * p * r + kappa * (1. - p), 0.)
        return np.r_[grad_r.ravel(), grad_p.ravel()]

    def dist(self, i, j):
        p = self.p[i, j]
        r = self.r[i, j]
        return stats.nbinom(r, 1. - p)


class GammaKernel(object):
    def __init__(self, alpha0, beta0, offset=1e-5):
        self.n_params = alpha0.size + beta0.size
        self._theta_alpha = np.log(alpha0)
        self._theta_beta = np.log(beta0)
        self.offset = offset

    @property
    def alpha(self):
        return np.exp(self._theta_alpha)

    @property
    def beta(self):
        return np.exp(self._theta_beta)

    def update(self, delta):
        n = self._theta_alpha.size
        self._theta_alpha += delta[:n]
        self._theta_beta += delta[n:]

    def likelihood(self, event_mark, dt):
        alpha = np.array(event_mark * self.alpha * event_mark.T)
        beta = np.array(event_mark * self.beta * event_mark.T)
        return np.where(dt >= 0, stats.gamma(a=alpha, scale=1. / beta).pdf(dt), 0.)

    def grad(self, event_mark, z, dt, n_observation, nu):
        _dt = np.maximum(dt, self.offset)
        n = n_observation + 1e-8
        alpha, beta = self.alpha, self.beta
        gamma = np.array(event_mark.T * (z * np.log(_dt)) * event_mark) / n
        kappa = np.array(event_mark.T * (z * _dt) * event_mark) / n
        ga = np.where(n_obseration > 0, alpha * (nu * (np.log(beta) - psi(alpha)) + gamma), 0.)
        gb = np.where(n_observation > 0, nu * alpha - kappa * beta, 0.)
        return np._r[ga.ravel(), gb.ravel()]

    def dist(self, i, j):
        alpha = self.alpha[i, j]
        beta = self.beta[i, j]        
        return stats.gamma(a=alpha, scale=1. / beta)
