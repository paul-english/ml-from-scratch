from unittest import TestCase

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from src.ml.decision_tree import DecisionTree
from src.utils.data import get_data
from src.utils.scoring import accuracy_score


class TestDecisionTree(TestCase):
    def test_initialization(self):
        clf = DecisionTree()

    def test_setting_a(self):
        X, y = get_data('mushroom', setting='SettingA')
        X_test, y_test = get_data('mushroom', setting='SettingA', test=True)

        clf = DecisionTree()
        clf.fit(X, y)
        y_hat = clf.predict(X_test)
        score = accuracy_score(y_test, y_hat)
        self.assertEqual(score, 1)

        # how does it compare to sklearn
        clf = DecisionTreeClassifier()
        Xt = pd.get_dummies(pd.DataFrame(X)).values
        Xt_test = pd.get_dummies(pd.DataFrame(X_test)).values
        clf.fit(Xt, y)
        y_hat = clf.predict(Xt_test)
        score = accuracy_score(y_test, y_hat)
        self.assertEqual(score, 1)

        # TODO test setting b & c

    def test_numerical_data(self):
        X, y = get_data('blobs')

        clf = DecisionTree()
        clf.fit(X, y)
        y_hat = clf.predict(X)
        score = accuracy_score(y, y_hat)
        self.assertTrue(score > 0.97)

    def test_numerical_data_works_with_missing_method_choices(self):
        X, y = get_data('blobs')

        clf = DecisionTree(missing_method='majority_value')
        clf.fit(X, y)
        y_hat = clf.predict(X)
        score = accuracy_score(y, y_hat)
        self.assertTrue(score > 0.97)

    def test_on_numeric_madelon(self):
        X, y = get_data('madelon')
        X_test, y_test = get_data('madelon', test=True)

        clf = DecisionTree()
        clf.fit(X, y)
        y_hat = clf.predict(X_test)
        score = accuracy_score(y_test, y_hat)
        self.assertTrue(score > 0.97)
