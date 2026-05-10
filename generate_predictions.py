"""
Generate predictions table with EV values for all matches.
"""
import pandas as pd
import numpy as np
import duckdb
from validation import train_lr, add_predictions, load_builtin, FEATURE_COLS

def calculate_ev(model_prob, odds_H, odds_D, odds_A, outcome_idx):
    """Calculate fair and margined EV for predicted outcome."""
    raw_odds = [odds_H, odds_D, odds_A][outcome_idx]
    implied_sum = (1/odds_H) + (1/odds_D) + (1/odds_A)
    
    fair_odds = raw_odds * implied_sum
    fair_ev = model_prob * fair_odds - 1
    margined_ev = model_prob * raw_odds - 1
    
    return fair_ev, margined_ev

def generate_predictions_table(league):
    """Generate predictions table for a league."""
    df = load_builtin(league)
    
    if df is None or len(df) == 0:
        return None
    
    train_seasons = sorted(df['season'].unique()[:-2])
    test_seasons = sorted(df['season'].unique()[-2:])
    
    df_train = df[df['season'].isin(train_seasons)].dropna(subset=FEATURE_COLS)
    df_test = df[df['season'].isin(test_seasons)].dropna(subset=['avg_odds_H'] + FEATURE_COLS)
    
    if len(df_test) == 0:
        return None
    
    lr_model, lr_scaler = train_lr(df_train)
    preds = add_predictions(df, df_test, lr_model, lr_scaler)
    
    if len(preds) == 0:
        return None
    
    results = []
    for _, r in preds.iterrows():
        odds_H = r['odds_H']
        odds_D = r['odds_D']
        odds_A = r['odds_A']
        implied_sum = (1/odds_H) + (1/odds_D) + (1/odds_A)
        
        lr_probs = r['lr']
        poisson_probs = r['poi']
        
        lr_fair = [lr_probs[0] * odds_H * implied_sum - 1,
                   lr_probs[1] * odds_D * implied_sum - 1,
                   lr_probs[2] * odds_A * implied_sum - 1]
        lr_margined = [lr_probs[0] * odds_H - 1,
                       lr_probs[1] * odds_D - 1,
                       lr_probs[2] * odds_A - 1]
        
        poisson_fair = [poisson_probs[0] * odds_H * implied_sum - 1,
                        poisson_probs[1] * odds_D * implied_sum - 1,
                        poisson_probs[2] * odds_A * implied_sum - 1]
        poisson_margined = [poisson_probs[0] * odds_H - 1,
                          poisson_probs[1] * odds_D - 1,
                          poisson_probs[2] * odds_A - 1]
        
        lr_pred = np.argmax(lr_fair)
        poisson_pred = np.argmax(poisson_fair)
        
        results.append({
            'date': r['match_date'],
            'home': r['home'],
            'away': r['away'],
            'actual': r['actual'],
            'lr_pred': ['H', 'D', 'A'][lr_pred],
            'lr_prob_H': lr_probs[0],
            'lr_prob_D': lr_probs[1],
            'lr_prob_A': lr_probs[2],
            'lr_fair_ev': lr_fair[lr_pred],
            'lr_margined_ev': lr_margined[lr_pred],
            'poisson_pred': ['H', 'D', 'A'][poisson_pred],
            'poisson_prob_H': poisson_probs[0],
            'poisson_prob_D': poisson_probs[1],
            'poisson_prob_A': poisson_probs[2],
            'poisson_fair_ev': poisson_fair[poisson_pred],
            'poisson_margined_ev': poisson_margined[poisson_pred],
            'odds_H': odds_H,
            'odds_D': odds_D,
            'odds_A': odds_A
        })
    
    return pd.DataFrame(results)

def main():
    conn = duckdb.connect('database/results.db')
    
    for league in ['PL', 'LL', 'SA', 'POL']:
        df = generate_predictions_table(league)
        
        if df is not None and len(df) > 0:
            table_name = f"{league}_predictions"
            conn.execute(f"DROP TABLE IF EXISTS {table_name}")
            conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM df")
    
    conn.close()

if __name__ == "__main__":
    main()