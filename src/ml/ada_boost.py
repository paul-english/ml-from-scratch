from __future__ import division, print_function

import copy

import numpy as np

from utils.scoring import accuracy_score


# https://cseweb.ucsd.edu/~yfreund/papers/IntroToBoosting.pdf

class AdaBoost(object):
    # TODO hyperparams for sub classifiers...
    @staticmethod
    def parameters():
        return {
            'T': [1, 5, 10, 20, 50, 100],
        }

    def __init__(self, classifiers, T=100):
        self.classifiers = classifiers
        self.T = T

    def fit(self, X, y):
        m, _ = X.shape
        self.D = np.full(m, 1 / m)
        self.h = [None for t in range(self.T)]
        self.alpha = np.zeros(self.T)

        for h in self.classifiers:
            h.fit(X, y)

        for t in range(self.T):
            min_error = 1
            for i,h in enumerate(self.classifiers):
                y_hat = h.predict(X)
                epsilon = 1/2 - 1/2*(self.D * y * h.predict(X)).sum()
                if epsilon < min_error:
                    min_error = epsilon
                    self.h[t] = h

            self.alpha[t] = 1/2 * np.log((1-min_error)/min_error)
            self.D = (self.D * np.exp(-self.alpha[t] * y * self.h[t].predict(X)))
            self.D /= self.D.sum() # Normalization, e.g. Z_t

        self.alpha /= self.alpha.sum()

    def predict(self, X):
        h = np.array([h.predict(X) for h in self.h])
        return np.sign(self.alpha.dot(h))

class AdaBoostRegressor(AdaBoost):

    @staticmethod
    def parameters():
        return {
            'loss': ['linear', 'square', 'exponential'],
            'T': [1, 5, 10, 20, 50, 100],
        }

    def __init__(self, classifiers, T=100, loss='linear'):
        self.classifiers = classifiers
        self.T = T
        self.loss = loss

    def _loss(self, y, y_hat):
        d = np.abs(y_hat - y)
        D = np.max(d) # not the same as self.D
        if self.loss == 'linear':
            return d / D
        elif self.loss == 'square':
            return d**2 / D
        elif self.loss == 'exponential':
            return 1 - np.exp(-d / D)

    def fit(self, X, y):
        m, _ = X.shape
        self.D = np.full(m, 1 / m)
        self.h = [None for t in range(self.T)]
        self.alpha = np.zeros(self.T)

        for h in self.classifiers:
            h.fit(X, y)

        for t in range(self.T):
            min_error = None
            for i, h in enumerate(self.classifiers):
                y_hat = h.predict(X)
                epsilon = self._loss(y, y_hat)
                avg_loss = epsilon.dot(self.D)
                if min_error is None or avg_loss < min_error:
                    min_error = avg_loss
                    self.h[t] = h

            if min_error > 0.5:
                self.alpha[t] = 0
                continue

            beta = min_error / (1 - min_error)
            self.alpha[t] = 1/2 * np.log((1-min_error)/min_error)
            self.D = self.D * beta**(1-min_error)
            self.D /= self.D

        self.alpha /= self.alpha.sum()

    def predict(self, X):
        h = np.array([h.predict(X) for h in self.h])
        return self.alpha.dot(h)
