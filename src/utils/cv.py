import numpy as np


def k_fold(y, cv=5):
    shuffled_idx = np.random.permutation(len(y))
    return np.array_split(shuffled_idx, cv)

def cross_validate(X, y, cv=5):
    for k, idx in enumerate(k_fold(y, cv=cv)):
        X_train, y_train = X[~idx], y[~idx]
        X_validate, y_validate = X[idx], y[idx]
        yield (X_train, y_train, X_validate, y_validate)
