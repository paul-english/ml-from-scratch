from unittest import TestCase

import numpy as np
from numpy.testing import assert_almost_equal

from src.utils.scoring import accuracy_score, f1_score, precision, recall


class TestScoring(TestCase):
    def test_accuracy(self):
        score = accuracy_score(
            np.array([0,0,0,0,0,1,1,1,1,1]),
            np.array([1,0,1,0,1,0,1,0,1,0]),
        )
        assert_almost_equal(score, 0.4)

    def test_precision(self):
        score = precision(
            np.array([0,0,0,0,0,1,1,1,1,1]),
            np.array([1,0,1,0,1,0,1,0,1,0]),
        )
        assert_almost_equal(score, 0.5)

    def test_recall(self):
        score = recall(
            np.array([0,0,0,0,0,1,1,1,1,1]),
            np.array([1,0,1,0,1,0,1,0,1,0]),
        )
        assert_almost_equal(score, 0.4)

    def test_f1_score(self):
        score = f1_score(
            np.array([0,0,0,0,0,1,1,1,1,1]),
            np.array([1,0,1,0,1,0,1,0,1,0]),
        )
        assert_almost_equal(score, 0.444444444)
