from __future__ import division, print_function

from pprint import pprint

import click

from corgi.ml import scores
from models import get_model
from utils.data import get_data
from utils.plotting import plot_nll


@click.command()
@click.option("--model", default='svm')
@click.option("--data", default='madelon')
@click.option("--params", default=None)
def main(model, data, params):
    X, y, metadata = get_data(data)
    X_test, y_test, _ = get_data(data, test=True)
    print('Data shape, Train: %s, Test: %s' % (X.shape, X_test.shape))

    if params is not None:
        params = eval(params)
    else:
        params = {}

    model_name = model
    model = get_model(model, metadata['regression'])

    params['metadata'] = metadata

    clf = model(**params)
    clf.fit(X, y)

    if hasattr(clf, '_negative_log_likelihood'):
        plot_nll(clf, model_name, data)

    y_pred = clf.predict(X)
    print('Train:')
    pprint(scores(y_pred, y, scoring=metadata['scoring']))

    y_pred = clf.predict(X_test)
    print('Test:')
    pprint(scores(y_pred, y_test, scoring=metadata['scoring']))

if __name__ == "__main__":
    main()
