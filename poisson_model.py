import os
import duckdb
import pandas as pd
import numpy as np
from scipy.stats import poisson

DB_PATH = os.path.join("database", "results.db")
LEAGUES = ["PL", "SA", "LL"]

def get_connection():
    return duckdb.connect(DB_PATH)

def get_team_stats(df_matches, team, is_home, n_matches=10):
    """Get average goals scored/conceded for a team in home/away games."""
    if is_home:
        home_games = df_matches[df_matches['home_team'] == team].sort_values('match_date').tail(n_matches)
        if len(home_games) == 0:
            return 1.0, 1.0
        scored = home_games['home_goals'].mean()
        conceded = home_games['away_goals'].mean()
    else:
        away_games = df_matches[df_matches['away_team'] == team].sort_values('match_date').tail(n_matches)
        if len(away_games) == 0:
            return 1.0, 1.0
        scored = away_games['away_goals'].mean()
        conceded = away_games['home_goals'].mean()
    
    return scored if not np.isnan(scored) else 1.0, conceded if not np.isnan(conceded) else 1.0

def predict_poisson(home_team, away_team, df_matches, n_matches=10, max_goals=8):
    """Predict match using Poisson model."""
    home_scored_home, home_conceded_home = get_team_stats(df_matches, home_team, is_home=True, n_matches=n_matches)
    
    away_scored_away, away_conceded_away = get_team_stats(df_matches, away_team, is_home=False, n_matches=n_matches)
    
    lambda_home = (home_scored_home + away_conceded_away) / 2
    lambda_away = (away_scored_away + home_conceded_home) / 2
    
    home_probs = [poisson.pmf(k, lambda_home) for k in range(max_goals + 1)]
    away_probs = [poisson.pmf(k, lambda_away) for k in range(max_goals + 1)]
    
    p_home_win = 0
    p_draw = 0
    p_away_win = 0
    
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            if i > j:
                p_home_win += home_probs[i] * away_probs[j]
            elif i == j:
                p_draw += home_probs[i] * away_probs[j]
            else:
                p_away_win += home_probs[i] * away_probs[j]
    
    total = p_home_win + p_draw + p_away_win
    if total > 0:
        p_home_win /= total
        p_draw /= total
        p_away_win /= total
    
    return {
        'lambda_home': lambda_home,
        'lambda_away': lambda_away,
        'P(H)': p_home_win,
        'P(D)': p_draw,
        'P(A)': p_away_win
    }

def validate_poisson(league, train_seasons, test_seasons, n_matches, max_goals):
    conn = get_connection()
    
    train_str = ",".join([f"'{s}'" for s in train_seasons])
    test_str = ",".join([f"'{s}'" for s in test_seasons])
    
    df_all = conn.execute(f"""
        SELECT match_date, home_team, away_team, home_goals, away_goals, result, season
        FROM matches_raw
        WHERE league = '{league}'
        ORDER BY match_date
    """).df()
    
    # Get test matches only
    df_test = df_all[df_all['season'].isin(test_seasons)].copy()
    
    results = []
    
    for idx, row in df_test.iterrows():
        df_prior = df_all[df_all['match_date'] < row['match_date']].copy()
        
        if len(df_prior) < 5:
            continue
            
        pred = predict_poisson(row['home_team'], row['away_team'], df_prior, n_matches=n_matches, max_goals=max_goals)
        
        actual_result = row['result']
        if actual_result == 'H':
            actual = 0
        elif actual_result == 'D':
            actual = 1
        else:
            actual = 2
        
        results.append({
            'home': row['home_team'],
            'away': row['away_team'],
            'actual': actual,
            'P(H)': pred['P(H)'],
            'P(D)': pred['P(D)'],
            'P(A)': pred['P(A)'],
            'lambda_home': pred['lambda_home'],
            'lambda_away': pred['lambda_away']
        })
    
    conn.close()
    
    df_res = pd.DataFrame(results)
    
    if len(df_res) == 0:
        return None
    
    y_true = df_res['actual'].values
    y_prob = df_res[['P(H)', 'P(D)', 'P(A)']].values
    
    from sklearn.metrics import log_loss
    logloss = log_loss(y_true, y_prob)
    
    preds = np.argmax(y_prob, axis=1)
    accuracy = (preds == y_true).mean()
    
    return {
        'league': league,
        'n_matches': len(df_res),
        'logloss': logloss,
        'accuracy': accuracy,
        'df': df_res
    }

def test_params():
    train_seasons = ["1617", "1718", "1819", "1920", "2021", "2122", "2223", "2324"]
    test_seasons = ["2425", "2526"]
    
    print("Testing Poisson model with different parameters:")
    print("=" * 60)
    
    for n_matches in [5, 10, 15]:
        for max_goals in [6, 8, 10]:
            print(f"\nn_matches={n_matches}, max_goals={max_goals}")
            
            for league in LEAGUES:
                result = validate_poisson(league, train_seasons, test_seasons, n_matches, max_goals)
                if result:
                    print(f"  {league}: logloss={result['logloss']:.4f}, acc={result['accuracy']:.1%}")

if __name__ == "__main__":
    pass