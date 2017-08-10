import numpy as np

import dill
from decision_tree import DecisionTree
from joblib import Parallel, delayed
from linear_models.svm import SVM


def build_predict_tree(k, max_depth, m, X, y):
    tree = DecisionTree(
        sample_features=k,
        max_depth=max_depth,
        best_attr_method='gini'
    )
    sample = np.random.choice(X.shape[0], m, replace=True)
    tree.fit(X[sample], y[sample])
    #self.trees.append(tree)
    #D.append(tree.predict(X))
    return tree

class SVMRandomForest(object):

    @staticmethod
    def parameters():
        return {
            'N': [5, 10, 25],
            'm': [32, 64, 128],

            # RF hyper params
            'max_depth': [1, 2, 3, 4],

            # SVM hyper params
            'C': [0.001, 0.01, 0.1, 1.0, 2.0, 10.0],
            'gamma': [0.001, 0.01, 0.1, 0.5, 1.0],
        }

    def __init__(self, m=100, N=5,
                 max_depth=5,
                 C=1.0,
                 gamma=0.01,
                 intercept=True,
                 weight_initialization='random',
                 shuffle=True):
        self.m = m
        self.N = N

        self.max_depth = max_depth

        self.C = C
        self.gamma = gamma
        self.intercept = intercept
        self.weight_initialization = weight_initialization
        self.shuffle = shuffle

    def fit(self, X, y):
        self.trees = []
        self.k = int(np.log2(X.shape[1]).round())

        D = []
        self.trees = Parallel(n_jobs=8)(
            delayed(build_predict_tree)(self.k, self.max_depth, self.m, X, y)
            for n in range(self.N)
        )
        D = [tree.predict(X) for tree in self.trees]
        #self.trees = [r[1] for r in results]

        # for n in range(self.N):
        #     tree = DecisionTree(
        #         sample_features=self.k,
        #         max_depth=self.max_depth,
        #         best_attr_method='gini'
        #     )
        #     sample = np.random.choice(X.shape[0], self.m, replace=True)
        #     tree.fit(X[sample], y[sample])
        #     self.trees.append(tree)
        #     D.append(tree.predict(X))
        D = np.array(D).T

        self.svm = SVM(
            C=self.C,
            gamma=self.gamma,
            intercept=self.intercept,
            weight_initialization=self.weight_initialization,
            shuffle=self.shuffle,
        )
        self.svm.fit(D, y)

    def predict(self, X):
        D = []
        for n in range(self.N):
            D.append(self.trees[n].predict(X))
        D = np.array(D).T
        return self.svm.predict(D)
