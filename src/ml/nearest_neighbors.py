from __future__ import division, print_function

import numpy as np

from scipy.spatial import cKDTree


def euclidean_distance(a, b):
    return np.sqrt(((a - b)**2).sum())

def manhattan_distance(a, b):
    return np.abs(a - b).sum()

class NearestNeighbors(object):
    @staticmethod
    def parameters():
        return {
            'k': [3, 5, 7, 9],
            'distance_weighted': [True, False],
        }

    def __init__(self, k=3, distance_weighted=True):
        self.k = k
        self.distance_weighted = distance_weighted

    def fit(self, X, y):
        self._y = y
        self.kd_tree = cKDTree(X, leafsize=10)

    def predict(self, X):
        predictions = []
        for element in X:
            d, i = self.kd_tree.query(element, self.k, n_jobs=-1)
            #print('--- d, i', d, i)
            neighbor_y = self._y[i]
            if self.distance_weighted:
                w = 1/((d+1)**2)
                w /= w.sum()
                #print('--- w', w, neighbor_y.dot(w).sum(), neighbor_y.mean())
                predictions.append(np.sign(neighbor_y.dot(w).sum()))
            else:
                predictions.append(np.sign(neighbor_y.mean()))
        return np.array(predictions)
