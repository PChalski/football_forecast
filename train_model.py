import os
import duckdb
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss
from sklearn.metrics import brier_score_loss

DB_PATH = os.path.join("database", "football.db")
LEAGUES = ["PL", "SA", "LL"]

def get_connection():
    return duckdb.connect(DB_PATH)

def encode_result(result):
    if result == 'H':
        return 0
    elif result == 'D':
        return 1
    else:
        return 2

def train_and_evaluate(league, train_seasons, test_seasons):
    conn = get_connection()
    
    train_str = ",".join([f"'{s}'" for s in train_seasons])
    test_str = ",".join([f"'{s}'" for s in test_seasons])

    query = f"""
        SELECT f.match_date, f.home_team, f.away_team, f.result, f.season,
               f.form_diff_5, f.form_diff_10, f.form_diff_15,
               f.gd_diff_5, f.gd_diff_10, f.gd_diff_15,
               f.home_advantage,
               f.home_away_form_diff, f.league_pos_diff,
               f.implied_H, f.implied_D, f.implied_A,
               e.elo_prob_H, e.elo_prob_D, e.elo_prob_A
        FROM features_{league} f
        LEFT JOIN elo_ratings e 
            ON f.match_date = e.match_date 
            AND f.home_team = e.home_team 
            AND f.away_team = e.away_team
        WHERE f.season IN ({train_str}, {test_str})
        ORDER BY f.match_date
    """
    df = conn.execute(query).df()
    conn.close()

    df_train = df[df['season'].isin(train_seasons)].copy()
    df_test = df[df['season'].isin(test_seasons)].copy()

    feature_cols = ['form_diff_5', 'form_diff_10', 'form_diff_15',
                  'gd_diff_5', 'gd_diff_10', 'gd_diff_15',
                  'home_advantage', 'home_away_form_diff', 'league_pos_diff']

    df_train = df_train.dropna(subset=['form_diff_5', 'gd_diff_5'])
    df_test = df_test.dropna(subset=['form_diff_5', 'gd_diff_5'])

    X_train = df_train[feature_cols].values
    y_train = df_train['result'].apply(encode_result).values
    X_test = df_test[feature_cols].values
    y_test = df_test['result'].apply(encode_result).values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(
        solver='lbfgs',
        max_iter=2000,
        C=0.5
    )
    model.fit(X_train_scaled, y_train)

    proba_model = model.predict_proba(X_test_scaled)
    logloss_model = log_loss(y_test, proba_model)
    
    brier_H = brier_score_loss(y_test == 0, proba_model[:, 0])
    brier_D = brier_score_loss(y_test == 1, proba_model[:, 1])
    brier_A = brier_score_loss(y_test == 2, proba_model[:, 2])
    brier_mean = (brier_H + brier_D + brier_A) / 3

    book_probs = np.array([
        df_test['implied_H'].values,
        df_test['implied_D'].values,
        df_test['implied_A'].values
    ]).T
    book_probs = np.nan_to_num(book_probs, nan=1/3)
    book_probs = book_probs / book_probs.sum(axis=1, keepdims=True)
    logloss_buk = log_loss(y_test, book_probs)
    
    brier_buk_H = brier_score_loss(y_test == 0, book_probs[:, 0])
    brier_buk_D = brier_score_loss(y_test == 1, book_probs[:, 1])
    brier_buk_A = brier_score_loss(y_test == 2, book_probs[:, 2])
    brier_buk_mean = (brier_buk_H + brier_buk_D + brier_buk_A) / 3

    elo_probs = np.array([
        df_test['elo_prob_H'].values,
        df_test['elo_prob_D'].values,
        df_test['elo_prob_A'].values
    ]).T
    elo_probs = np.nan_to_num(elo_probs, nan=1/3)
    elo_probs = elo_probs / elo_probs.sum(axis=1, keepdims=True)
    logloss_elo = log_loss(y_test, elo_probs)
    
    brier_elo_H = brier_score_loss(y_test == 0, elo_probs[:, 0])
    brier_elo_D = brier_score_loss(y_test == 1, elo_probs[:, 1])
    brier_elo_A = brier_score_loss(y_test == 2, elo_probs[:, 2])
    brier_elo_mean = (brier_elo_H + brier_elo_D + brier_elo_A) / 3

    print(f"\n{league}:")
    print(f"  Train: {len(df_train)}, Test: {len(df_test)}")
    print(f"  Model LR:  log-loss = {logloss_model:.4f}, brier = {brier_mean:.4f}")
    print(f"  Bukmacher: log-loss = {logloss_buk:.4f}, brier = {brier_buk_mean:.4f}")
    print(f"  ELO:       log-loss = {logloss_elo:.4f}, brier = {brier_elo_mean:.4f}")
    print(f"  vs Buk:     log-loss {logloss_model - logloss_buk:+.4f}, brier {brier_mean - brier_buk_mean:+.4f}")
    print(f"  vs ELO:     log-loss {logloss_model - logloss_elo:+.4f}, brier {brier_mean - brier_elo_mean:+.4f}")

if __name__ == "__main__":
    train_seasons = ["1617", "1718", "1819", "1920", "2021", "2122", "2223", "2324"]
    test_seasons = ["2425", "2526"]

    print("Training and validation")
    print(f"Train: {train_seasons}")
    print(f"Test:  {test_seasons}")
    print("="*50)

    for league in LEAGUES:
        train_and_evaluate(league, train_seasons, test_seasons)