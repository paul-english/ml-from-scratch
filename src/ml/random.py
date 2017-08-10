import numpy as np


class RandomNormalRegressor(object):
    @staticmethod
    def parameters():
        return {}

    def __init__(self, **kwargs):
        pass

    def fit(self, X, y):
        self.mu = np.mean(y)
        self.std = np.std(y)

    def predict(self, X):
        return np.random.normal(self.mu, self.std, size=len(X))
