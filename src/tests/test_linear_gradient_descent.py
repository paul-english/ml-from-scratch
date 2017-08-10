from unittest import TestCase

from numpy.testing import assert_almost_equal

from src.ml.linear_gradient_descent import LinearGradientDescent
from src.utils.data import get_data
from src.utils.scoring import accuracy_score


class TestLinearGradientDescent(TestCase):
    def test_initialization(self):
        clf = LinearGradientDescent()

    def test_fit(self):
        clf = LinearGradientDescent()
        X, y = get_data('table2')
        clf.fit(X, y)
        y_hat = clf.predict(X)
        score = accuracy_score(y, y_hat)
        assert_almost_equal(score, 0.833333333)
