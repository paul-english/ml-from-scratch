from ml.ada_boost import AdaBoost
from ml.decision_tree import DecisionTree
from ml.external.keras_wrapper import WrappedKerasRegressor
from ml.external.xgboost_wrapper import WrappedXGBRegressor
from ml.linear_models.lasso import LassoRegression, PairwiseLassoRegression
from ml.linear_models.linear_gradient_descent import (LinearGradientDescent,
                                                      LinearGradientDescentRegressor)
from ml.linear_models.linear_least_squares import (LinearLeastSquares,
                                                   LinearLeastSquaresRegressor)
from ml.linear_models.linear_stochastic_gradient_descent import (LinearStochasticGradientDescent,
                                                                 LinearStochasticGradientDescentRegressor,
                                                                 PairwiseLinearRegression)
from ml.linear_models.logistic_regression import LogisticRegression
from ml.linear_models.perceptron import Perceptron
from ml.linear_models.poisson import (PairwisePoissonRegression,
                                      PoissonRegression)
from ml.linear_models.ridge import (PairwiseRidgeRegression, RidgeRegression,
                                    RidgeRegressionGradient)
from ml.linear_models.svm import SVM
from ml.listnet import ListNetRegression
from ml.naive_bayes import NaiveBayes
from ml.nearest_neighbors import NearestNeighbors
from ml.random import RandomNormalRegressor
from ml.random_forest import SVMRandomForest

regression_models = {
    # External ml algorithms
    'xgboost': WrappedXGBRegressor,
    'keras': WrappedKerasRegressor,

    # Homebrew ml algorithms
    'gradient_descent': LinearGradientDescentRegressor,
    'least_squares': LinearLeastSquaresRegressor,
    'stochastic_gradient_descent': LinearStochasticGradientDescentRegressor,
    'lasso': LassoRegression,
    'ridge': RidgeRegression,
    'ridge_gd': RidgeRegressionGradient,
    'poisson': PoissonRegression,

    'pairwise_linear': PairwiseLinearRegression,
    'pairwise_lasso': PairwiseLassoRegression,
    'pairwise_ridge': PairwiseRidgeRegression,
    'pairwise_poisson': PairwisePoissonRegression,

    'listnet': ListNetRegression,

    'random': RandomNormalRegressor,
}

classifier_models = {
    # Homebrew ml algorithms
    'ada_boost': AdaBoost,
    'decision_tree': DecisionTree,
    'gradient_descent': LinearGradientDescent,
    'least_squares': LinearLeastSquares,
    'stochastic_gradient_descent': LinearStochasticGradientDescent,
    'logistic': LogisticRegression,
    'perceptron': Perceptron,
    'svm_random_forest': SVMRandomForest,
    'svm': SVM,
    'naive_bayes': NaiveBayes,
    'knn': NearestNeighbors,
}

def get_model(name, is_regression):
    if is_regression:
        model = regression_models.get(name)
        if model is None:
            raise Exception("Model %s not found for regression" % name)
    else:
        model = classifier_models.get(name)
        if model is None:
            raise Exception("Model %s not found for classification" % name)

    return model
