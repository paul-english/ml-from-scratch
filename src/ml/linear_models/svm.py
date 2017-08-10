from __future__ import division, print_function

import numpy as np


# TODO use base stochastic

class SVM(object):

    @staticmethod
    def parameters():
        return {
            'C': [0.001, 0.01, 0.1, 1.0, 2.0, 10.0],
            'gamma': [0.001, 0.01, 0.1, 0.5, 0.9, 1.0],
            'intercept': [True, False],
            'epochs': [5],
            'weight_initialization': ['random', 'zeros'],
            'shuffle': [True, False],
        }

    def __init__(self,
                 C=1.0,
                 gamma=0.01,
                 intercept=True,
                 epochs=20,
                 weight_initialization='zeros',
                 shuffle=True):
        self.C = C
        self._gamma = gamma
        self.intercept = intercept
        self.epochs = epochs
        self.weight_initialization = weight_initialization
        self.all_weight_magnitudes = []
        self.all_gamma = []
        self.t = 0
        self.shuffle = shuffle

    def _width(self, X):
        if self.intercept:
            return X.shape[1] + 1
        else:
            return X.shape[1]

    def _weights(self, n):
        if self.weight_initialization == 'random':
            self.w = np.random.normal(0, 1, n)
        elif self.weight_initialization == 'zeros':
            self.w = np.zeros(n)
        else:
            raise Exception('Unknown weight initialization method: %s' % (self.weight_initialization,))

    def gamma(self):
        return self._gamma / (1 + (self._gamma * (self.t / self.C)))

    def _add_intercept(self, X):
        dim = X.shape[1] if len(X.shape) == 2 else X.shape[0]
        if self.intercept and dim == self._dim:
            X = np.hstack([
                X, np.full([X.shape[0], 1], 1.0)
            ])
        return X

    def fit(self, X, y):
        self._weights(self._width(X))
        self._dim = X.shape[1]

        for n in range(self.epochs):
            if self.shuffle:
                shuffle_idx = np.random.permutation(X.shape[0])
                self.fit_partial(X[shuffle_idx], y[shuffle_idx])
            else:
                self.fit_partial(X, y)

    def fit_partial(self, X, y):
        if self.w is None:
            raise Exception('Model must be fit first')

        X = self._add_intercept(X)

        for X_i, y_i in zip(X, y):
            gamma = self.gamma()

            self.w *= (1 - gamma)

            if y_i * self.w.dot(X_i) <= 1:
                self.w += gamma * self.C * y_i * X_i

            self.all_weight_magnitudes.append(np.linalg.norm(self.w))
            self.all_gamma.append(gamma)
            self.t += 1

    def predict(self, X):
        if self.w is None:
            raise Exception('Model must be fit first')

        X = self._add_intercept(X)
        return np.sign(np.dot(self.w, X.T))

    # TODO would like to be able to plot it's performance automagically..
