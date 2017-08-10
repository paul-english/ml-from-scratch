from unittest import TestCase

import numpy as np

from src.ml.ada_boost import AdaBoost
from src.ml.decision_tree import DecisionTree
from src.ml.rule_classifier import RuleClassifier
from src.utils.data import get_data
from src.utils.scoring import accuracy_score


class TestAdaBoost(TestCase):
    def test_initialization(self):
        clf = AdaBoost([DecisionTree()])

    def test_classifier(self):
        classifiers=[
            RuleClassifier(lambda x: np.sign(1.5 - x[:, 0])),
            RuleClassifier(lambda x: np.sign(4.5 - x[:, 0])),
            RuleClassifier(lambda x: np.sign(x[:, 1] - 5)),
        ]
        clf = AdaBoost(classifiers, T=5)
        X = np.array([
            [1, 2],
            [1, 4],
            [2.5, 5.5],
            [3.5, 6.5],
            [4, 5.4],
            [2, 1],
            [2, 4],
            [3.5, 3.5],
            [5, 2],
            [5, 5.5],
        ])
        y = np.array([1,1,1,1,1,-1,-1,-1,-1,-1])
        clf.fit(X, y)
        y_hat = clf.predict(X)
        score = accuracy_score(y, y_hat)
        self.assertEqual(score, 1)

    def test_hw_example(self):
        classifiers = [
            RuleClassifier(lambda x: np.sign(x[:,0])),
            RuleClassifier(lambda x: np.sign(x[:,0] - 2)),
            RuleClassifier(lambda x: -np.sign(x[:,0])),
            RuleClassifier(lambda x: -np.sign(x[:,1])),
        ]
        clf = AdaBoost(classifiers, T=4)
        X = np.array([
            [1,1],
            [1,-1],
            [-1,-1],
            [-1,1],
        ])
        y = np.array([-1,1,-1,-1])
        clf.fit(X, y)
        y_hat = clf.predict(X)
        score = accuracy_score(y, y_hat)
        self.assertEqual(score, 1)

    def test_stuff(self):
        return # TODO

        # TODO how do you use the same base classifier over & over?
        # nothing makes it any different, so how do you get it
        # to select different things...
        X,y = get_data('table2')
        clf = AdaBoost([
            DecisionTree(max_depth=1),
            DecisionTree(max_depth=2),
            DecisionTree(max_depth=3),
            DecisionTree(max_depth=1, best_attr_method='gini'),
            DecisionTree(max_depth=2, best_attr_method='gini'),
            DecisionTree(max_depth=3, best_attr_method='gini'),
        ], T=5)
        clf.fit(X, y)
        y_hat = clf.predict(X)
        score = accuracy_score(y, y_hat)
        self.assertEqual(score, 1)
