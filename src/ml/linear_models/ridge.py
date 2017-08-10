from __future__ import division, print_function

import itertools

import numpy as np

from base import BaseStochastic
from tqdm import tqdm


class RidgeRegressionGradient():
    def __init__(self, r=0.001, gamma=0.01, epochs=5):
        self.r = r
        self._gamma = gamma
        self.epochs = epochs

    def gamma(self, t):
        return self._gamma / (1 + (self._gamma * (t / self.r)))

    def fit(self, X, y):
        self.w = np.random.normal(0,0.1,size=X.shape[1])
        for t in range(self.epochs):
            J = 2*(X.dot(self.w)-y).dot(X) + (2*self.gamma(t)*self.w)
            self.w -= self.r * J

    def predict(self, X):
        return X.dot(self.w)


class RidgeRegression(BaseStochastic):
    @staticmethod
    def parameters():
        parameters = BaseStochastic.parameters()
        parameters['r'] = [0.001, 0.01, 0.1, 1.0]
        parameters['gamma'] = [0.001, 0.01, 0.1, 1.0]
        return parameters

    def __init__(self, r=0.01, gamma=0.01, **kwargs):
        self.r = r
        self._gamma = gamma
        super(RidgeRegression, self).__init__(**kwargs)

    def gamma(self):
        return self._gamma / (1 + (self._gamma * (self.t / self.r)))

    def fit_partial(self, X, y):
        if self.w is None:
            raise Exception('Model must be fit first')
        X = self._add_intercept(X)
        for X_i, y_i in zip(X, y):
            J = 2*(X_i.dot(self.w)-y_i)*X_i + (2*self.gamma()*self.w)
            self.w -= self.r * J
            self.t += 1


class PairwiseRidgeRegression(BaseStochastic):
    @staticmethod
    def parameters():
        parameters = BaseStochastic.parameters()
        parameters['r'] = [0.001, 0.01, 0.1]
        parameters['gamma'] = [0.01, 0.1, 1.0]
        return parameters

    def __init__(self, r=0.001, gamma=0.01, **kwargs):
        self.r = r
        self._gamma = gamma
        super(PairwiseRidgeRegression, self).__init__(**kwargs)

    def gamma(self):
        return self._gamma / (1 + (self._gamma * (self.t / self.r)))

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

                    J = 2*(X_i.dot(self.w)-y_i)*X_i + (2*self.gamma()*self.w)

                    self.w += self.r * J
                    self.t += 1
                    pbar.update()

    def fit_partial(self, X, y):
        raise NotImplementedError()
