from unittest import TestCase

from numpy.testing import assert_almost_equal

from src.ml.perceptron import Perceptron
from src.utils.data import get_data
from src.utils.scoring import accuracy_score


class TestPerceptron(TestCase):
    def test_initialization(self):
        clf = Perceptron()

    def test_fit(self):
        clf = Perceptron(weight_initialization='zeros', shuffle=False)
        X, y = get_data('table2')
        clf.fit(X, y)
        y_hat = clf.predict(X)
        score = accuracy_score(y, y_hat)
        assert_almost_equal(score, 0.8333333)
