
import numpy as np


class AdaGrad(object):
    def __init__(self, alpha=.1):
        '''Trainer class for AdaGrad algorithm.'''
        self.alpha = alpha
        self.reset()
        
    def reset(self):
        self._v = 0.
        
    def update(self, g):
        self._v += g**2
        return self.alpha * g / (np.sqrt(self._v) + 1e-8)
        
class Adam(object):
    def __init__(self, alpha=.1, beta1=.9, beta2=.999, epsilon=1e-8):
        '''Trainer class for Adam algorithm.'''
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.reset()
        
    def reset(self):
        self._m = 0.
        self._v = 0.
        self._t = 0.
        
    def update(self, g):
        b1, b2 = self.beta1, self.beta2
        self._t += 1
        self._m = b1 * self._m + (1. - b1) * g
        self._v = b2 * self._v + (1. - b2) * g**2
        alpha = self.alpha * np.sqrt(1. - b2**self._t) / (1. - b1**self._t)
        return alpha * self._m / (np.sqrt(self._t * self._v) + self.epsilon)