from __future__ import division, print_function

import numpy as np


class Perceptron(object):

    @staticmethod
    def parameters():
        return {
            'r': [0.001, 0.01, 0.1, 0.2, 0.5, 0.9, 1.0],
            'intercept': [True, False],
            'epochs': [5],
            'mu': [0, 0.5, 1.0, 2.5, 5.0],
            'weight_initialization': ['random', 'zeros'],
            'shuffle': [True, False],
            'update_method': ['simple', 'aggressive'],
        }

    def __init__(self,
                 r=0.1,
                 intercept=True,
                 epochs=10,
                 mu=0,
                 verbose=False,
                 weight_initialization='random',
                 shuffle=True,
                 update_method='simple'):
        self.r = r
        self.intercept = intercept
        self.epochs = epochs
        self.mu = mu
        self.verbose = verbose
        self.weight_initialization = weight_initialization
        self.shuffle = shuffle
        self.update_method = update_method

        self.mistakes = 0
        self._classes = []

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

    def _update(self, X, y):
        if self.update_method == 'aggressive':
            eta = (self.mu - (y * np.dot(self.w, X))) / (np.dot(X.T, X) + 1)
            return eta * (y * X)
        elif self.update_method == 'simple':
            return self.r * (y * X)
        else:
            raise Exception('Unknown update method: %s' % (self.update_method,))

    def fit(self, X, y):
        self._weights(self._width(X))
        self._dim = X.shape[1]
        self.mistakes = 0

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
            if y_i * (np.dot(self.w, X_i)) <= self.mu:
                self.mistakes += 1
                self.w += self._update(X_i, y_i)

    def predict(self, X):
        if self.w is None:
            raise Exception('Model must be fit first')

        X = self._add_intercept(X)
        return np.sign(np.dot(self.w, X.T))
