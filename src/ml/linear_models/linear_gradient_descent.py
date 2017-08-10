from __future__ import division, print_function

import numpy as np


class LinearGradientDescent(object):

    @staticmethod
    def parameters():
        return {
            'T': [1, 5, 10, 20, 50, 100],
            'r': [0.001, 0.01, 0.1, 0.2, 0.5, 0.9, 1.0],
        }

    def __init__(self, T=10, r=0.01):
        self.T = T
        self.r = r

    def fit(self, X, y):
        _,m = X.shape
        self.w = np.zeros(m)
        for t in range(self.T):
            J = (X.dot(self.w) - y).dot(X)
            self.w -= self.r * J

    def predict(self, X):
        return np.sign(X.dot(self.w))

class LinearGradientDescentRegressor(LinearGradientDescent):

    def predict(self, X):
        return X.dot(self.w)
