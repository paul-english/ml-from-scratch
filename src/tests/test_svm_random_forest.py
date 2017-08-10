from unittest import TestCase

import numpy as np
from numpy.testing import assert_almost_equal

from src.ml.random_forest import SVMRandomForest
from src.utils.data import get_data
from src.utils.scoring import accuracy_score


class TestSVMRandomForest(TestCase):
    def test_initialization(self):
        clf = SVMRandomForest()

    def test_fit_handwriting(self):
        np.random.seed(0)
        clf = SVMRandomForest(N=5, m=100)
        X, y = get_data('handwriting')
        clf.fit(X, y)
        y_hat = clf.predict(X)
        score = accuracy_score(y, y_hat)
        assert_almost_equal(score, 1)
        # TODO

    def test_fit_madelon(self):
        np.random.seed(0)
        clf = SVMRandomForest(N=5, m=20)
        X, y = get_data('madelon')
        clf.fit(X, y)
        y_hat = clf.predict(X)
        score = accuracy_score(y, y_hat)
        assert_almost_equal(score, -1)
        # TODO
