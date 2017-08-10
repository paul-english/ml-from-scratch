from __future__ import division, print_function

import numpy as np

from base import BaseStochastic
from scipy.misc import factorial


class PoissonRegression(BaseStochastic):
    @staticmethod
    def parameters():
        parameters = BaseStochastic.parameters()
        parameters['r'] = [0.001, 0.01, 0.1, 0.2, 0.5, 0.9, 1.0]
        parameters['gamma'] = [0.001, 0.01, 0.1, 0.2, 0.5, 0.9, 1.0],
        return parameters

    def __init__(self, r=0.01, gamma=0.01, **kwargs):
        self.r = r
        self._gamma = gamma
        super(PoissonRegression, self).__init__(**kwargs)

    def gamma(self):
        # TODO... allow fixed gamma
        #return self._gamma
        return self._gamma / (1 + (self._gamma * (self.t / self.r)))

    def fit_partial(self, X, y):
        if self.w is None:
            raise Exception('Model must be fit first')
        X = self._add_intercept(X)
        for X_i, y_i in zip(X, y):
            # TODO
            print('----', X_i.dot(self.w))
            J = -np.exp(X_i.dot(self.w) + 1) * X_i + (X_i * y_i)
            #J = -np.exp(X_i.dot(self.w)) + (X_i.dot(self.w) * y_i)
            #J = -X_i * (y_i - X_i.dot(self.w)) + self.gamma() * np.abs(self.w).sum()
            #J = -self.w + X_i * np.log(self.w) - np.log(factorial(X_i))
            self.w += self.r * J
            print('--- J', J)
            print('--- w', self.w)
            break
            self.t += 1

class PairwisePoissonRegression(object):
    pass
