from __future__ import division, print_function

import numpy as np

import pandas as pd
from corgi.ml import classifier_scoring, rank_scoring, regression_scoring
from corgi.stats import boxcox
from sklearn.datasets.samples_generator import (make_blobs, make_circles,
                                                make_moons, make_regression)
from utils.nfl_data import nfl_player_data, nfl_team_data


def k_fold(y, cv=5):
    shuffled_idx = np.random.permutation(y)
    return np.array_split(shuffled_idx, cv)

def read_libsvm(file_loc):
    data = []
    y = []
    max_idx = 0

    # read all input data into list of dicts
    with open(file_loc, 'r') as f:
        for line in f.readlines():
            pieces = line.split(' ')
            y.append(int(pieces[0]))
            row = {}
            for piece in pieces[1:]:
                if ':' not in piece:
                    continue
                idx, val = piece.split(':')
                idx, val = int(idx), int(val)
                if idx > max_idx:
                    max_idx = idx
                row[idx] = val
            data.append(row)

    # convert list of sparse dicts into dense numpy array
    X = np.zeros([len(data), max_idx])
    for i, row in enumerate(data):
        for j in range(max_idx):
            X[i, j] = row.get(j, 0)

    y = np.array(y)

    return X, y

def _get_mushroom_data(setting=None, test=False):
    if setting is None:
        setting = 'SettingA'
    if test:
        filename = 'test'
    else:
        filename = 'training'
    loc = 'data/mushroom/%s/%s.data' % (setting, filename)
    data = np.genfromtxt(loc, delimiter=',', dtype=None)

    metadata = {
        'regression': False,
        'scoring': classifier_scoring,
        'primary_metric': 'accuracy',
    }

    return data[:,:-1], data[:,-1], metadata

def _get_adult_data(test=False):
    if test:
        X, y = read_libsvm('data/adult/a5a.test')
        # the test set has an extra column that we can't capture
        # in training, we just delete it.
        X = np.delete(X, 122, 1)
    else:
        X, y = read_libsvm('data/adult/a5a.train')

    metadata = {
        'regression': False,
        'scoring': classifier_scoring,
        'primary_metric': 'accuracy',
    }

    return X, y, metadata

def _get_table2_test_data():
    X, y = read_libsvm('data/adult/table2')

    metadata = {
        'regression': False,
        'scoring': classifier_scoring,
        'primary_metric': 'accuracy',
    }

    return X, y, metadata

def _get_handwriting_data(test=False):
    if test:
        filename = 'test'
    else:
        filename = 'train'
    data = np.genfromtxt('data/handwriting/%s.data' % filename, delimiter=' ', dtype='b')
    labels = np.genfromtxt('data/handwriting/%s.labels' % filename, delimiter=' ', dtype=None)

    metadata = {
        'regression': False,
        'scoring': classifier_scoring,
        'primary_metric': 'accuracy',
    }

    return data, labels, metadata

def _get_madelon_data(test=False):
    if test:
        filename = 'test'
    else:
        filename = 'train'
    data = np.genfromtxt('data/madelon/madelon_%s.data' % filename, delimiter=' ', dtype=None)
    labels = np.genfromtxt('data/madelon/madelon_%s.labels' % filename, delimiter=' ', dtype=None)

    metadata = {
        'regression': False,
        'scoring': classifier_scoring,
        'primary_metric': 'accuracy',
    }

    return data, labels, metadata

def _get_nfl_team_data(test=False, trim_before=2014, season_kind='Regular'):
    team_data = nfl_team_data(trim_before=trim_before, season_kind=season_kind)
    team_data = team_data.dropna(axis=0, subset=['next_game_team_score'])

    most_recent_game = team_data[team_data.start_time == team_data.start_time.max()].iloc[0]
    test_idx = (team_data.season_year == most_recent_game.season_year) & \
               (team_data.season_type == most_recent_game.season_type) & \
               (team_data.season_week == most_recent_game.season_week)

    if season_kind is not None:
        team_data = team_data.drop(['season_type'], axis=1)

    x_cols = [c for c in team_data.columns]
    x_cols = [c for c in x_cols if not c.startswith('next_game')]
    x_cols = [c for c in x_cols if c not in ['start_time', 'gsis_id']]

    X = pd.get_dummies(team_data[x_cols])
    y = team_data.next_game_team_score

    X = X.fillna(0)
    X = X.astype(np.float32)

    # normalize
    X /= np.linalg.norm(X)

    metadata = {
        'regression': True,
        'scoring': rank_scoring,
        'columns': list(X.columns),
        'primary_metric': 'endcg',
    }

    if test:
        return X[test_idx].values, y[test_idx].values, metadata
    else:
        return X[~test_idx].values, y[~test_idx].values, metadata

def _get_nfl_player_data(test=False, trim_before=2014, season_kind='Regular'):
    team_data = nfl_team_data(trim_before=trim_before, season_kind=season_kind)
    team_data = team_data.dropna(axis=0, subset=['next_game_team_score'])
    player_data = nfl_player_data(team_data, trim_before=trim_before, season_kind=season_kind)
    player_data = player_data.dropna(axis=0, subset=['next_game_score'])

    # TODO classify unk positions on what we think they are

    # Only look at relevant player positions
    player_data = player_data[player_data.position.isin([
        'QB', 'RB', 'WR', 'TE',
    ])]

    most_recent_game = player_data[player_data.start_time == player_data.start_time.max()].iloc[0]
    test_idx = (player_data.season_year == most_recent_game.season_year) & \
               (player_data.season_type == most_recent_game.season_type) & \
               (player_data.season_week == most_recent_game.season_week)

    if season_kind is not None:
        player_data = player_data.drop(['season_type'], axis=1)

    x_cols = [c for c in player_data.columns]
    x_cols = [c for c in x_cols if not c.startswith('next_game')]
    x_cols = [c for c in x_cols if c not in ['start_time', 'gsis_id']]

    X = pd.get_dummies(player_data[x_cols])
    y = player_data.next_game_score

    X = X.fillna(0)
    X = X.astype(np.float32)

    # normalize
    X /= np.linalg.norm(X)

    # Player scores are heavily skewed, we use a boxcox transform to
    # address this. We don't need to worry about inversing this, since
    # it's a monotonic transformation, and if we predict a player has
    # a higher boxcox score, they would likewise have a higher score.
    y, ld, offset = boxcox(y)
    y.index = X.index # so that we can still use test_idx
    print('Boxcox parameters: ld %s, offset %s' % (ld, offset))

    metadata = {
        'regression': True,
        'scoring': rank_scoring,
        'columns': list(X.columns),
        'primary_metric': 'endcg',
    }

    if test:
        return X[test_idx].values, y[test_idx].values, metadata
    else:
        return X[~test_idx].values, y[~test_idx].values, metadata

def _get_moons(*args, **kwargs):
    X, y = make_moons(n_samples=100, noise=0.1)
    metadata = {
        'regression': False,
        'scoring': classifier_scoring,
        'primary_metric': 'accuracy',
    }
    return X, y, metadata

def _get_circles(*args, **kwargs):
    X, y = make_circles(n_samples=100, noise=0.1)
    metadata = {
        'regression': False,
        'scoring': classifier_scoring,
        'primary_metric': 'accuracy',
    }
    return X, y, metadata

def _get_blobs(*args, **kwargs):
    X, y = make_blobs(n_samples=100, centers=2, n_features=3)
    metadata = {
        'regression': False,
        'scoring': classifier_scoring,
        'primary_metric': 'accuracy',
    }
    return X, y, metadata

def _get_line(*args, **kwargs):
    X, y = make_regression(
        n_samples=200,
        n_features=3,
        n_informative=3,
        bias=20,
        noise=0.2,
        random_state=19522
    )

    if kwargs.get('test'):
        X, y = X[:100], y[:100]
    else:
        X, y = X[100:], y[100:]

    metadata = {
        'regression': True,
        'scoring': regression_scoring,
        'primary_metric': 'mean_squared_error',
    }
    return X, y, metadata

def get_data(name, *args, **kwargs):
    data_fns = {
        'adult': _get_adult_data,
        'mushroom': _get_mushroom_data,
        'table2': _get_table2_test_data,
        'handwriting': _get_handwriting_data,
        'madelon': _get_madelon_data,
        'moons': _get_moons,
        'circles': _get_circles,
        'blobs': _get_blobs,
        'nfl_team': _get_nfl_team_data,
        'nfl_player': _get_nfl_player_data,
        'line': _get_line,
    }
    data_fn = data_fns.get(name)
    if data_fn is None:
        raise Exception("Unknown data set")

    X, y, metadata = data_fn(*args, **kwargs)

    return X, y, metadata
