from __future__ import division, print_function

import numpy as np

from scipy.linalg import qr


class LinearLeastSquares(object):

    @staticmethod
    def parameters():
        return {}

    def __init__(self):
        pass
    # TODO allow intercepts

    def fit(self, X, y):
        # Not efficient or recommended, but the easiest implementation
        self.w = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
        print('--- w', self.w)

    def predict(self, X):
        return np.sign(X.dot(self.w))

class LinearLeastSquaresRegressor(LinearLeastSquares):

    def predict(self, X):
        return X.dot(self.w)
