from __future__ import division, print_function

import numpy as np


class LogisticRegression(object):

    @staticmethod
    def parameters():
        return {
            'r': [0.001, 0.01, 0.1, 0.2, 0.5, 0.9, 1.0],
            'sigma': [0.01, 0.1, 1.0, 5.0, 10.0, 20.0],
            'intercept': [True, False],
            'epochs': [5],
            'weight_initialization': ['random', 'zeros'],
            'shuffle': [True, False],
        }

    def __init__(self,
                 r=0.01,
                 sigma=0.1,
                 intercept=True,
                 epochs=20,
                 weight_initialization='zeros',
                 shuffle=True):
        self.r = r
        self.sigma = sigma
        self.intercept = intercept
        self.epochs = epochs
        self.weight_initialization = weight_initialization
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
        self._negative_log_likelihood = []

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
            J = -y_i * X_i.dot(1 / (1 + np.exp(y_i * X_i.dot(self.w)))) \
                + (2*self.w)/(self.sigma**2)
            self.w -= self.r * J

        negative_log_likelihood = (self.w.dot(self.w)/(self.sigma**2) + np.log(1 + np.exp(-y * X.dot(self.w)))).sum()
        self._negative_log_likelihood.append(negative_log_likelihood)

    def predict(self, X):
        if self.w is None:
            raise Exception('Model must be fit first')

        X = self._add_intercept(X)

        # for the {1,-1} classification case, we don't need to
        # use the sigmoid function. which will incorrectly
        # scope the output to {0,1}, we can instead just use the
        # sign function since the weights represent a perfectly
        # valid linear classification
        return np.sign(X.dot(self.w))
