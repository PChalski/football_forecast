"""
Betting simulation module for Streamlit app.
"""
import pandas as pd
import numpy as np
import duckdb

def run_simulation(df, model, odds_type='margined', min_ev=0):
    """Run simulation on DataFrame with predictions."""
    if model not in df.columns:
        return None
    
    bankroll = 1000
    results = []
    
    for _, r in df.iterrows():
        probs = r[model]
        if not isinstance(probs, (list, np.ndarray)):
            continue
        
        pred = np.argmax(probs)
        model_prob = probs[pred]
        
        odds_H = r['odds_H']
        odds_D = r['odds_D']
        odds_A = r['odds_A']
        implied_sum = (1/odds_H) + (1/odds_D) + (1/odds_A)
        
        bet_odds = [odds_H, odds_D, odds_A][pred]
        if odds_type == 'fair':
            bet_odds = bet_odds * implied_sum
        
        ev = model_prob * bet_odds - 1
        
        if ev < min_ev:
            results.append({
                'bankroll': bankroll,
                'bet': 0,
                'ev': ev,
                'won': None
            })
            continue
        
        actual = r['actual']
        won = pred == actual
        
        ret = bet_odds if won else 0
        bankroll += ret - 1
        
        results.append({
            'bankroll': bankroll,
            'bet': 1,
            'return': ret,
            'ev': ev,
            'won': won
        })
    
    return pd.DataFrame(results)

def simulate_league(league, model, odds_type='margined', min_ev=0):
    """Run simulation for a specific league and model."""
    conn = duckdb.connect('database/results.db')
    
    table_name = f"{league}_predictions"
    tables = [r[0] for r in conn.execute("SHOW TABLES").fetchall()]
    
    if table_name not in tables:
        conn.close()
        return None
    
    preds = conn.execute(f"SELECT * FROM {table_name} ORDER BY date").df()
    conn.close()
    
    if len(preds) == 0:
        return None
    
    preds['actual'] = preds['actual'].astype(int)
    
    if model == 'LR':
        preds['model_probs'] = preds[['lr_prob_H', 'lr_prob_D', 'lr_prob_A']].values.tolist()
    elif model == 'Poisson':
        preds['model_probs'] = preds[['poisson_prob_H', 'poisson_prob_D', 'poisson_prob_A']].values.tolist()
    elif model == 'Random':
        use_random = True
    else:
        return None
    
    np.random.seed(42)
    bankroll = 1000
    results = []
    
    for _, r in preds.iterrows():
        odds_H = r['odds_H']
        odds_D = r['odds_D']
        odds_A = r['odds_A']
        implied_sum = (1/odds_H) + (1/odds_D) + (1/odds_A)
        
        if odds_type == 'fair':
            odds_fair = [odds_H * implied_sum, odds_D * implied_sum, odds_A * implied_sum]
        else:
            odds_fair = [odds_H, odds_D, odds_A]
        
        if model == 'Random':
            best_idx = np.random.choice([0, 1, 2])
            probs = [1/3, 1/3, 1/3]
        else:
            probs = r['model_probs']
            if odds_type == 'fair':
                evs = [probs[0] * odds_fair[0] - 1, probs[1] * odds_fair[1] - 1, probs[2] * odds_fair[2] - 1]
            else:
                evs = [probs[0] * odds_H - 1, probs[1] * odds_D - 1, probs[2] * odds_A - 1]
            best_idx = np.argmax(evs)
        
        best_ev = probs[best_idx] * odds_fair[best_idx] - 1
        
        if best_ev < min_ev:
            results.append({
                'bankroll': bankroll,
                'bet': 0,
                'ev': best_ev,
                'won': None
            })
            continue
        
        actual = r['actual']
        won = best_idx == actual
        
        bet_odds = odds_fair[best_idx] if odds_type == 'fair' else [odds_H, odds_D, odds_A][best_idx]
        
        ret = bet_odds if won else 0
        bankroll += ret - 1
        
        results.append({
            'bankroll': bankroll,
            'bet': 1,
            'return': ret,
            'ev': best_ev,
            'won': won
        })
    
    return pd.DataFrame(results)

def get_summary(sim_df):
    """Calculate summary statistics from simulation."""
    if sim_df is None or len(sim_df) == 0:
        return {}
    
    bets = sim_df[sim_df['bet'] == 1]
    
    if len(bets) == 0:
        return {
            'n_bets': 0,
            'total_return': 0,
            'profit': 0,
            'roi': 0,
            'hit_rate': 0,
            'final_bankroll': sim_df['bankroll'].iloc[-1] if len(sim_df) > 0 else 1000
        }
    
    final_bankroll = sim_df['bankroll'].iloc[-1]
    profit = final_bankroll - 1000
    roi = profit / 1000 * 100
    
    won_bets = bets[bets['won'].notna()]
    hit_rate = won_bets['won'].mean() * 100 if len(won_bets) > 0 else 0
    
    return {
        'n_bets': len(bets),
        'total_return': round(bets['return'].sum(), 2),
        'profit': round(profit, 2),
        'roi': round(roi, 2),
        'hit_rate': round(hit_rate, 2),
        'final_bankroll': round(final_bankroll, 2)
    }