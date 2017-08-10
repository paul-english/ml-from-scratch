from __future__ import division, print_function

import numpy as np


class BaseStochastic(object):
    @staticmethod
    def parameters():
        return {
            'intercept': [True, False],
            'epochs': [5],
            'weight_initialization': ['random', 'zeros'],
            'shuffle': [True, False],
        }

    def __init__(self, intercept=True, epochs=20, weight_initialization='zeros', shuffle=True, **kwargs):
        self.intercept = intercept
        self.epochs = epochs
        self.weight_initialization = weight_initialization
        self.shuffle = shuffle
        self.t = 0

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
        #print('-w', self.w)

    def predict(self, X):
        if self.w is None:
            raise Exception('Model must be fit first')

        X = self._add_intercept(X)
        return np.dot(self.w, X.T)
