
from __future__ import division, print_function


def accuracy_score(y, y_hat):
    return (y == y_hat).sum() / len(y)

def precision(y, y_hat):
    tp = ((y == 1) & (y_hat == 1)).sum()
    fp = ((y == -1) & (y_hat == 1)).sum()
    return tp / (tp + fp)

def recall(y, y_hat):
    tp = ((y == 1) & (y_hat == 1)).sum()
    fn = ((y == 1) & (y_hat == -1)).sum()
    return tp / (tp + fn)

def f1_score(y, y_hat):
    p = precision(y, y_hat)
    r = recall(y, y_hat)
    return 2 * ((p*r) / (p+r))
