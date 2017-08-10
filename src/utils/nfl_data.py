import os
from datetime import datetime, timedelta

import numpy as np
import pytz

import pandas as pd
from db import get_aleph, get_nfldb, get_session

dir_path = os.path.dirname(os.path.realpath(__file__))

with open(os.path.join(dir_path, '../sql/team_data.sql'), 'r') as f:
    TEAM_DATA_SQL = f.read()
with open(os.path.join(dir_path, '../sql/player_data.sql'), 'r') as f:
    PLAYER_DATA_SQL = f.read()

player_scoring = {
    'passing_yds': 0.04,
    'passing_tds': 4,
    'passing_int': -1,
    'rushing_yds': 0.1,
    'rushing_tds': 6,
    'receiving_rec': 0.5,
    'receiving_yds': 0.1,
    'receiving_tds': 6,
    'kickret_tds': 6,
    'puntret_tds': 6,
    'fumbles_lost': -2,
    'passing_twoptm': 2,
    'rushing_twoptm': 2,
    'receiving_twoptm': 2,
}

team_scoring = {
    'defense_sk': 1,
    'defense_safe': 2,
    'defense_int': 2,
    'defense_frec': 2,
    'defense_fgblk': 2,
    'defense_puntblk': 2,
    'defense_int_tds': 6,
    'defense_misc_tds': 6,
    'defense_frec_tds': 6,
}

def pts_allowed(x):
    if x == 0: return 10
    elif x <= 6: return 7
    elif x <= 13: return 4
    elif x <= 20: return 1
    elif x <= 27: return 0
    elif x <= 34: return -1
    else: return -4

def team_defense_score(df):
    df['team_defense_score'] = np.dot(
        df[team_scoring.keys()],
        team_scoring.values()
    )
    df['team_defense_score'] += df.apply(lambda x: pts_allowed(x.team_score), axis=1)
    defense_scores = pd.Series(df.team_defense_score.values, index=df.team).to_dict()
    df['opponent_defense_score'] = df.apply(lambda x: defense_scores[x.opponent], axis=1)
    return df

def player_score(df):
    df['score'] = np.dot(
        df[player_scoring.keys()],
        player_scoring.values()
    )
    return df

def shift_by_game(df, shift_col, old_col, new_col, shift_by):
    df[new_col] = df.sort('start_time', ascending=True)[
        [shift_col, old_col]
    ].groupby(
        [shift_col]
    ).shift(shift_by)
    return df

def tz_offset(tz1, tz2):
    now = datetime.now() # could use any time, it's relative to itself. There's probably a better way to get this
    return (tz1.utcoffset(now).seconds // 3600) - (tz2.utcoffset(now).seconds // 3600)


def nfl_team_data(trim_before=None, season_kind='Regular'):
    nfldb_engine = get_nfldb()

    where = ''
    if trim_before is not None:
        where += ' AND game.season_year >= %s' % trim_before

    if season_kind is not None:
        where += ' AND game.season_type = \'%s\'' % season_kind

    # TODO filter by player position, e.g. we only want to look at players in the predicted positions
    teams = pd.read_sql(TEAM_DATA_SQL % where, nfldb_engine)

    # Generate Fantasy Football scores using known ff rules
    teams = team_defense_score(teams)

    # Moving the current weeks score to last week since we want to use current weeks
    # data to predict the next week.
    teams = shift_by_game(teams, 'team', 'team_score', 'next_game_team_score', -1)
    teams = shift_by_game(teams, 'team', 'team_defense_score', 'next_game_team_defense_score', -1)
    teams = shift_by_game(teams, 'opponent', 'opponent_score', 'next_game_opponent_score', -1)
    teams = shift_by_game(teams, 'opponent', 'opponent_defense_score', 'next_game_opponent_defense_score', -1)

    # Pull past week information forward so we can use it for baselines
    # and as predictors
    teams = shift_by_game(teams, 'team', 'team_score', 'last_game_team_score', 1)
    teams = shift_by_game(teams, 'team', 'team_defense_score', 'last_game_team_defense_score', 1)
    teams = shift_by_game(teams, 'opponent', 'opponent_score', 'last_game_opponent_score', 1)
    teams = shift_by_game(teams, 'opponent', 'opponent_defense_score', 'last_game_opponent_defense_score', 1)

    # Rolling average features
    teams['rolling_team_score_5'] = teams.team_score.rolling(window=5).mean()
    teams['rolling_team_score_10'] = teams.team_score.rolling(window=10).mean()
    teams['rolling_team_defense_score_5'] = teams.team_defense_score.rolling(window=5).mean()
    teams['rolling_team_defense_score_10'] = teams.team_defense_score.rolling(window=10).mean()
    teams['rolling_opponent_score_5'] = teams.opponent_score.rolling(window=5).mean()
    teams['rolling_opponent_score_10'] = teams.opponent_score.rolling(window=10).mean()
    teams['rolling_opponent_defense_score_5'] = teams.opponent_defense_score.rolling(window=5).mean()
    teams['rolling_opponent_defense_score_10'] = teams.opponent_defense_score.rolling(window=10).mean()

    # Timezone offset between away team and home team.
    # Assuming that if the away team is heavily offset
    # their performance could be worse
    team_timezones = pd.read_csv('data/nfl_team_timezones.csv')
    team_tz_lookup = {
        team.team_id: pytz.timezone(team.tz_name)
        for _, team in team_timezones.iterrows()
    }
    teams['team_offset'] = teams.apply(lambda x: tz_offset(
        team_tz_lookup.get(x.team),
        team_tz_lookup.get(x.opponent)
    ), axis=1)

    return teams

def nfl_player_data(teams_df, trim_before=None, season_kind='Regular'):
    # TODO want to incorporate the teams df
    nfldb_engine = get_nfldb()

    where = ""
    if trim_before is not None:
        where += ' AND game.season_year >= %s' % trim_before

    if season_kind is not None:
        where += ' AND game.season_type = \'%s\'' % season_kind

    players = pd.read_sql(PLAYER_DATA_SQL % where, nfldb_engine)

    # Generate Fantasy Football scores using known ff rules
    players = player_score(players)

    # Moving the current weeks score to last week since we want to use current weeks
    # data to predict the next week.
    players = shift_by_game(players, 'player_id', 'score', 'next_game_score', -1)

    # Moving the current weeks score to last week since we want to use current weeks
    # data to predict the next week.
    players = shift_by_game(players, 'player_id', 'score', 'last_game_score', 1)

    # Rolling average features
    players['rolling_score_5'] = players.score.rolling(window=5).mean()
    players['rolling_score_10'] = players.score.rolling(window=10).mean()

    #print('--- player positions', np.unique(players.position))
    #print('--- player positions', players.position.value_counts())
    # TODO classify player positions?

    return players
