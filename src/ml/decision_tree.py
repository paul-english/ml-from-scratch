from __future__ import division, print_function

import itertools
import sys
from collections import Counter, namedtuple

import numpy as np


def counts(x):
    return np.array(Counter(x).values())


def _entropy(p):
    n = p.sum()
    proportion = p / n
    return np.nan_to_num(-proportion * np.log2(proportion)).sum()


def _gini(p):
    n = p.sum()
    proportion = p / n
    return (proportion * (1 - proportion)).sum()


def entropy(S):
    return _entropy(counts(S))


def gini(S):
    return _gini(counts(S))


def gain(S, A, measure=entropy):
    n = len(S)
    summands = np.array([
        len(A[A == x]) * measure(S[A == x])
        for x in np.unique(A)
        if len(S[A == x]) > 0
    ]) / n
    return measure(S) - sum(summands)


def accuracy_score(y, y_hat):
    return (y == y_hat).sum() / len(y)


def most_common_value(x, regression=False):
    if regression or (is_continuous(x) and len(x) == len(np.unique(x))):
        return np.mean(x)
    c = Counter(x)
    return c.keys()[np.argmax(c.values())]


def most_common_attr_value(X, a):
    attrs = X[:, a]
    attrs = attrs[attrs != '?']
    return most_common_value(attrs)


def _categorical_split(S, attribute, Y, method):
    if method == 'entropy':
        attr_gain = gain(Y, S[:, attribute])
    elif method == 'gini':
        attr_gain = gain(Y, S[:, attribute], measure=gini)

    split_values = np.unique(S[:,attribute])
    split_fn = lambda x: x

    return attr_gain, split_values, split_fn

def _continuous_split(S, attribute, Y, method, max_continous_splits_considered):
    x = S[:, attribute]
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = Y[sort_idx]

    y_value_change_indices = np.where(y_sorted[:-1] != y_sorted[1:])[0]
    if len(y_value_change_indices) > max_continous_splits_considered:
        y_value_change_indices = np.array([
            a[-1] for a in np.array_split(y_value_change_indices, max_continous_splits_considered+1)[:-1]
        ])

    gains = []
    for split in y_value_change_indices:
        x_split = np.ones(x.shape[0])
        x_split[split:] = 0
        if method == 'entropy':
            gains.append(gain(y_sorted, x_split))
        elif method == 'gini':
            gains.append(gain(y_sorted, x_split, measure=gini))

    best_split = y_value_change_indices[np.argmax(gains)]
    best_gain = np.max(gains)

    best_split_value = x_sorted[best_split]

    split_values = [True, False]
    split_fn = lambda x: x <= best_split_value

    return best_gain, split_values, split_fn

def is_continuous(X):
    numeric = np.issubdtype(X.dtype, np.number)
    boolean_only = np.in1d(np.unique(X),[0,1]).all()
    return numeric and not boolean_only

def best_split(S, a, Y, method, max_splits_considered):
    if is_continuous(S):
        gain, splits, split_fn = _continuous_split(S, a, Y, method, max_splits_considered)
    else:
        gain, splits, split_fn = _categorical_split(S, a, Y, method)

    return gain, a, splits, split_fn


Tree = namedtuple('Tree', 'attribute_index nodes split_fn')
Leaf = namedtuple('Leaf', 'label')


class DecisionTree(object):
    @staticmethod
    def parameters():
        return {
            'max_depth': [1, 2, 3, 4, 5, 10, 15],
            #'missing_method': ['majority_value', 'majority_label', 'special'],
            'best_attr_method': ['entropy', 'gini'],
            'sample_features': [2, 5, 10, 20, 50],
            # TODO good choice of k if it's none is np.log2(X.shape[1]).round()
       }

    def __init__(self,
                 verbose=0,
                 max_depth=10,
                 missing_method='special',
                 best_attr_method='entropy',
                 sample_features=20,
                 max_continous_splits_considered=4,
                 regression=False):

        self.verbose = verbose
        self.max_depth = max_depth
        self.depth = 0
        self.missing_method = missing_method
        self.best_attr_method = best_attr_method
        self.sample_features = sample_features
        self.max_continous_splits_considered = max_continous_splits_considered
        self.regression = regression

        # with Parallel(n_jobs=8) as parallel:
        #     self.parallel = parallel

    def _best_attribute(self, S, attributes, Y):
        if self.sample_features is not None:
            attrs = list(np.random.choice(list(attributes), int(self.sample_features), replace=False))
        else:
            attrs = list(attributes)

        splits = [
            best_split(S, a, Y, self.best_attr_method, self.max_continous_splits_considered)
            for a in attrs
        ]

        best_gain = 0
        best_attr = None
        best_splits = None
        best_split_fn = None

        for (gain, a, split_vals, split_fn) in splits:
            if (best_attr is None) or (gain >= best_gain):
                best_gain = gain
                best_attr = a
                best_splits = split_vals
                best_split_fn = split_fn

        if best_attr is None:
            raise Exception("Unable to find best split attribute for set: %s" % attrs)

        return best_attr, best_splits, best_split_fn

    def _id3(self, S, attributes, Y, depth=0):
        if depth > self.depth:
            self.depth = depth
        if depth >= self.max_depth:
            return Leaf(label=most_common_value(Y, self.regression))

        if np.all(Y[0] == Y):
            return Leaf(label=Y[0])

        if len(attributes) == 0:
            return Leaf(label=most_common_value(Y, self.regression))

        a, split_values, split_fn = self._best_attribute(S, attributes, Y)

        root = Tree(attribute_index=a, nodes={}, split_fn=split_fn)

        for value in split_values:
            Sv = S[split_fn(S[:,a]) == value]
            Yv = Y[split_fn(S[:,a]) == value]

            if Sv.shape[0] <= 1:
                root.nodes[value] = Leaf(label=most_common_value(Y, self.regression))
            else:
                if np.issubdtype(S.dtype, np.number):
                    root.nodes[value] = self._id3(Sv, attributes, Yv, depth+1)
                else:
                    root.nodes[value] = self._id3(Sv, (attributes - {a}), Yv, depth+1)

        return root

    def _populate_missing_attr_values(self, X, y):
        if np.issubdtype(X.dtype, np.number):
            pass
        elif self.missing_method == 'majority_value':
            self.majority_attr = {
                a: most_common_attr_value(X, a)
                for a in self._attributes
            }
        elif self.missing_method == 'majority_label':
            self.majority_attr = {
                a: most_common_attr_value(X, a)
                for a in self._attributes
            }
            self.majority_attr_given_label = {}
            label_attr = itertools.product(self._labels, self._attributes)
            for label, a in label_attr:
                self.majority_attr_given_label[(label, a)] = most_common_attr_value(X[y == label], a)
        elif self.missing_method == 'special':
            # The tree handles this by default, do nothing
            pass
        else:
            raise Exception('Unknown missing feature method.')

    def _coerce_missing_attr_values(self, X, y=None):
        if np.issubdtype(X.dtype, np.number):
            pass
        elif self.missing_method == 'majority_value':
            # replace with most common for each attribute
            for a in self._attributes:
                X[:,a][X[:,a] == '?'] = self.majority_attr[a]
        elif self.missing_method == 'majority_label':
            # replace with most common value given that label
            label_attr = itertools.product(self._labels, self._attributes)
            if y is not None:
                for label,a in label_attr:
                    X[:,a][(y==label) & (X[:,a] == '?')] = self.majority_attr_given_label[(label,a)]

            else:
                # this method seems like it's mostly nonsense, but....
                # "When you use method 2, for the test set unknown label, method 2 will degraded into "method 1", most frequent value of all." ~ Jie Cao
                for a in self._attributes:
                    X[:,a][X[:,a] == '?'] = self.majority_attr[a]
        elif self.missing_method == 'special':
            pass
        else:
            raise Exception('Unknown missing feature method.')
        return X

    def fit(self, X, y):
        X, y = X.copy(), y.copy()
        self._attributes = set(range(X.shape[1]))
        self._labels = np.unique(y)
        self._most_common_y = most_common_value(y, self.regression)

        self._populate_missing_attr_values(X, y)
        X = self._coerce_missing_attr_values(X, y)

        self._attr_values = {a: np.unique(X[:,a]) for a in self._attributes}

        self._tree = self._id3(X, self._attributes, y)

    def predict(self, X):
        X = X.copy()
        X = self._coerce_missing_attr_values(X)

        predictions = []

        for element in X:
            tree = self._tree

            while not isinstance(tree, Leaf):
                attribute_value = element[tree.attribute_index]
                node_value = tree.split_fn(attribute_value)
                tree = tree.nodes.get(
                    node_value,
                    Leaf(label=self._most_common_y)
                )

            predictions.append(tree.label)

        return np.array(predictions)

    def to_dot(self, tree=None, depth=0, parent=None, i=None, value=None):
        output = ""
        root = False

        if tree is None:
            root = True
            tree = self._tree

        if root and isinstance(tree, Leaf):
            node_id = 'root_leaf'
            output += 'graph G {\n'
            output += 'root_leaf [label="%s"];\n' % (tree.label)
        elif root:
            node_id = 'node_%s_%s' % (depth, tree.attribute_index)
            output += 'graph G {\n'
            output += '%s [label="%s"];\n' % (node_id, tree.attribute_index)
        elif isinstance(tree, Leaf):
            node_id = '%s_leaf_%s_%s_%s' % (parent, depth, tree.label, i)
            output += '%s [shape=box, label="%s"];\n' % (node_id, tree.label)
            output += '%s -- %s [label="%s"];\n' % (parent, node_id, value)
        else:
            node_id = '%s_node_%s_%s_%s' % (parent, depth, tree.attribute_index, i)
            output += '%s [label="%s"];\n' % (node_id, tree.attribute_index)
            output += '%s -- %s [label="%s"];\n' % (parent, node_id, value)

        if not isinstance(tree, Leaf):
            for i, (value, node) in enumerate(tree.nodes.iteritems()):
                output += self.to_dot(
                    tree=node,
                    depth=depth+1,
                    parent=node_id,
                    i=i,
                    value=value,
                )

        if root:
            output += '}\n'

        return output
