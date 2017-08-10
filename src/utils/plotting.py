from __future__ import division, print_function

import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


def plot_learning_curve(train_scores, test_scores, n, experiment, title):
    plt.figure()
    ax = plt.subplot(111)
    ax.set_title("Learning Curve - %s" % title)
    ax.set_xlabel("Training examples")
    ax.set_ylabel("Score")

    ax.plot(n, train_scores, 'o-', color="r", label="Cross-validation score")
    ax.plot(n, test_scores, 'o-', color="g", label="Test score")

    ax.legend(loc="best")

    plt.savefig("visualizations/%s_%s_learning_curve.png" % (experiment, title))
    return plt

def plot_nll(clf, model_name, data):
    plt.figure()
    ax = plt.subplot(111)
    ax.set_title("Negative Log Likelihood")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Negative Log Likelihood")
    plt.plot(clf._negative_log_likelihood)
    plt.savefig("visualizations/%s_%s_nll.png" % (model_name, data))
