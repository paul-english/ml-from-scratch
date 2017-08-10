from __future__ import division, print_function

import itertools

import click
import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from corgi.ml import scores
from models import get_model
from tqdm import tqdm
from utils.cv import cross_validate
from utils.data import get_data


@click.command()
@click.option("--model", default='svm')
@click.option("--data", default='madelon')
@click.option("--cv", default=5)
@click.option("--repetitions", default=3, help='Since many of the classifier runs have somewhat random results, this parameter repeats a CV run and averages together results.')
@click.option("--repeat-test", default=False, is_flag=True, help='Repeat the test model predictions and average the scores. Will repeat according to the `repetitions` parameter')
def main(model, cv, data, repetitions, repeat_test):
    X, y, metadata = get_data(data)
    X_test, y_test, _ = get_data(data, test=True)
    print('Data shape, Train: %s, Test: %s' % (X.shape, X_test.shape))
    print('CV %s, repetitions %s' % (cv, repetitions))

    model_name = model
    model = get_model(model, metadata['regression'])

    parameter_search = list(itertools.product(*model.parameters().values()))

    results = []

    model_args = list(model.parameters().keys())

    pbar = tqdm(parameter_search)
    best_cv_score = 0

    for parameters in pbar:
        model_kwargs = dict(zip(model_args, parameters))
        model_kwargs['metadata'] = metadata
        pbar.set_description("Best score: %s" % (best_cv_score,))
        #print('-', model_kwargs)
        clf = model(**model_kwargs)
        cv_scores = []
        for n in range(repetitions):
            for X_train, y_train, X_validate, y_validate in cross_validate(X, y, cv=cv):
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_validate)
                cv_scores.append(
                    scores(y_validate, y_pred, scoring=metadata['scoring'])
                )

        cv_scores = pd.DataFrame(cv_scores)
        score = {}
        for col in cv_scores.columns:
            score['cv_%s_mean'% col] = cv_scores[col].mean()
            score['cv_%s_std'% col] = cv_scores[col].std()
        score['model_kwargs'] = model_kwargs
        results.append(score)

        primary_score = score['cv_%s_mean' % metadata['primary_metric']]
        if primary_score > best_cv_score:
            best_cv_score = primary_score

    results = pd.DataFrame(results)
    results = results.join(pd.DataFrame(results["model_kwargs"].to_dict()).T)

    best_idx = results['cv_%s_mean' % metadata['primary_metric']].argmax()
    best_score = results.ix[best_idx]

    clf = model(**best_score.model_kwargs)
    clf.fit(X, y)
    y_pred = clf.predict(X_test)

    if repeat_test:
        print('-- test scores', scores(y_test, y_pred, scoring=metadata['scoring']))
        test_scores = dict(pd.DataFrame([
            scores(y_test, y_pred, scoring=metadata['scoring'])
            for i in range(repetitions)
        ]).mean(axis=0))
    else:
        test_scores = scores(y_test, y_pred, scoring=metadata['scoring'])

    print("Best score results:", best_score)

    # TODO q-q plot
    # TODO train/test plot

    name = '%s__%s__cv_%s__rep_%s' % (model_name, data, cv, repetitions)
    sns.pairplot(results[model_args + [c for c in results.columns if c.startswith('cv_')]])
    plt.savefig("visualizations/cv_pairplot__%s.png" % name)

    results.to_csv("results/scores/%s.csv" % name)

    print("-------------------")
    print("Test score:", test_scores)


if __name__ == "__main__":
    main()
