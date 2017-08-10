from __future__ import division, print_function

import itertools

import numpy as np

from base import BaseStochastic
from tqdm import tqdm


# TODO add in the same features as the SVM
class LinearStochasticGradientDescent(BaseStochastic):

    @staticmethod
    def parameters():
        parameters = BaseStochastic.parameters()
        parameters['r'] = [0.001, 0.01, 0.1, 1.0]
        return parameters

    def __init__(self, r=0.01, **kwargs):
        self.r = r
        super(LinearStochasticGradientDescent, self).__init__(**kwargs)

    def fit_partial(self, X, y):
        if self.w is None:
            raise Exception('Model must be fit first')
        X = self._add_intercept(X)
        for X_i, y_i in zip(X, y):
            J = (X_i.dot(self.w) - y_i) * X_i
            self.w -= self.r * J

    # TODO can abstract a lot of the linear model functions
    def predict(self, X):
        X = self._add_intercept(X)
        return np.sign(X.dot(self.w))

class LinearStochasticGradientDescentRegressor(LinearStochasticGradientDescent):

    def predict(self, X):
        print('---- w', self.w)
        X = self._add_intercept(X)
        return X.dot(self.w)

class PairwiseLinearRegression(BaseStochastic):
    @staticmethod
    def parameters():
        parameters = BaseStochastic.parameters()
        parameters['r'] = [0.001, 0.01, 0.1, 1.0]
        parameters['gamma'] = [0.001, 0.01, 0.1, 1.0]
        return parameters

    def __init__(self, r=0.001, **kwargs):
        self.r = r
        super(PairwiseLinearRegression, self).__init__(**kwargs)

    def fit(self, X, y):
        self._weights(self._width(X))
        self._dim = X.shape[1]

        X = self._add_intercept(X)

        total = (len(X)**2 - len(X)) * self.epochs
        with tqdm(total=total) as pbar:
            for n in range(self.epochs):
                for a,b in itertools.permutations(range(len(X)), 2):
                    X_i = X[a] - X[b]
                    y_i = y[a] - y[b]

                    J = (X_i.dot(self.w) - y_i) * X_i

                    self.w += self.r * J
                    self.t += 1
                    pbar.update()

    def fit_partial(self, X, y):
        raise NotImplementedError()
