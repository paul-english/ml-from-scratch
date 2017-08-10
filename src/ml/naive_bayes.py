import numpy as np

from scipy.stats import norm


class NaiveBayes(object):
    @staticmethod
    def parameters():
        return {
            'sigma_eps': [1e-3, 1e-2, 1e-1, 1.0, 2.0, 5.0, 10.0], # always smooth, we don't want 0 in the param
            'likelihood_eps': [0.0, 1e-5, 1e-2, 1e-1, 1.0, 2.0, 5.0, 10.0],
            'intercept': [True, False],
        }

    def __init__(self, sigma_eps=1.0, likelihood_eps=1e-2, intercept=True):
        # smoothing on the sigma parameter, having 0 here is bad,
        # and greater allowable deviance generally means we can fit
        # noisier data with better generality.
        self.sigma_eps = sigma_eps

        # smoothing on the product, just because
        self.likelihood_eps = likelihood_eps

        self.intercept = intercept

    def _add_intercept(self, X):
        dim = X.shape[1] if len(X.shape) == 2 else X.shape[0]
        if self.intercept and dim == self._dim:
            X = np.hstack([
                X, np.full([X.shape[0], 1], 1.0)
            ])
        return X

    def fit(self, X, y):
        self._dim = X.shape[1]
        X = self._add_intercept(X)

        self._classes = np.unique(y)
        self._p_x_given_y = [
            list(zip(
                X[y == label].mean(axis=0),
                X[y == label].std(axis=0)
            ))
            for label in self._classes
        ]
        # p, (1-p) for binomial classes
        self._p_y = [(y==v).mean() for v in self._classes]


    def predict(self, X):
        probs = self.predict_proba(X)
        return self._classes[probs.argmax(axis=1)]

    def predict_proba(self, X):
        X = self._add_intercept(X)
        return np.array([
            np.array([
                norm.logpdf(X[:, j], mu, sigma+self.sigma_eps) + self.likelihood_eps
                for j, (mu, sigma) in enumerate(p_x_given_y)
            ]).sum(axis=0) + np.log(p_y)
            for p_x_given_y, p_y in zip(self._p_x_given_y, self._p_y)
        ]).T
