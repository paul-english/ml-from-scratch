from unittest import TestCase

from numpy.testing import assert_almost_equal

from src.ml.svm import SVM
from src.utils.data import get_data
from src.utils.scoring import accuracy_score


class TestSVM(TestCase):
    def test_initialization(self):
        clf = SVM()

    def test_fit_handwriting(self):
        clf = SVM(shuffle=False, epochs=1, weight_initialization='zeros')
        X, y = get_data('handwriting')
        clf.fit(X, y)
        y_hat = clf.predict(X)
        score = accuracy_score(y, y_hat)
        assert_almost_equal(score, 0.927, decimal=4)

    def test_fit_madelon(self):
        clf = SVM(shuffle=False, epochs=1, weight_initialization='zeros')
        X, y = get_data('madelon')
        clf.fit(X, y)
        y_hat = clf.predict(X)
        score = accuracy_score(y, y_hat)
        assert_almost_equal(score, 0.5505, decimal=4)
