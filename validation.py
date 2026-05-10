from sklearn.metrics import log_loss, roc_auc_score
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import poisson

FEATURE_COLS = ['form_diff_5', 'form_diff_10', 'form_diff_15',
              'gd_diff_5', 'gd_diff_10', 'gd_diff_15',
              'home_advantage', 'home_away_form_diff', 'league_pos_diff']

DB_PATH = "database/results.db"

def get_team_stats(df, team, is_home, n_matches=20):
    if 'home_team' in df.columns:
        col = 'home_team' if is_home else 'away_team'
    else:
        col = 'home' if is_home else 'away'
    games = df[df[col] == team].sort_values('match_date').tail(n_matches)
    if len(games) == 0:
        return 1.0, 1.0
    
    scored_col = 'home_goals' if is_home else 'away_goals'
    conceded_col = 'away_goals' if is_home else 'home_goals'
    
    if scored_col not in games.columns:
        return 1.0, 1.0
    
    scored = games[scored_col].mean()
    conceded = games[conceded_col].mean()
    return (scored if not np.isnan(scored) else 1.0,
            conceded if not np.isnan(conceded) else 1.0)

def predict_poisson(home, away, df_prior, n_matches=20):
    hs, hc = get_team_stats(df_prior, home, True, n_matches)
    as_, ac = get_team_stats(df_prior, away, False, n_matches)
    lam_h = (hs + ac) / 2
    lam_a = (as_ + hc) / 2
    hp = [poisson.pmf(k, lam_h) for k in range(9)]
    ap = [poisson.pmf(k, lam_a) for k in range(9)]
    ph = pd_ = pa = 0.0
    for i in range(9):
        for j in range(9):
            if i > j: ph += hp[i] * ap[j]
            elif i == j: pd_ += hp[i] * ap[j]
            else: pa += hp[i] * ap[j]
    s = ph + pd_ + pa
    return np.array([ph / s, pd_ / s, pa / s])

def train_lr(df):
    df = df.dropna(subset=FEATURE_COLS)
    X = df[FEATURE_COLS].values
    y = df['result'].map({'H': 0, 'D': 1, 'A': 2}).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LogisticRegression(solver='lbfgs', max_iter=2000, C=0.5)
    model.fit(X_scaled, y)
    return model, scaler

def add_predictions(df_all, df_test, model_lr, scaler_lr):
    rows = []
    for _, r in df_test.iterrows():
        prior = df_all[df_all['match_date'] < r['match_date']]
        if len(prior) < 3:
            continue
        lr = model_lr.predict_proba(scaler_lr.transform([[r[f] for f in FEATURE_COLS]]))[0]
        poi = predict_poisson(r['home_team'], r['away_team'], prior)
        odds = np.array([r['avg_odds_H'], r['avg_odds_D'], r['avg_odds_A']])
        buk = 1 / odds
        buk /= buk.sum()
        rows.append({'match_date': r['match_date'], 'home': r['home_team'], 'away': r['away_team'],
                      'actual': {'H': 0, 'D': 1, 'A': 2}[r['result']],
                      'odds_H': odds[0], 'odds_D': odds[1], 'odds_A': odds[2],
                      'lr': lr, 'poi': poi, 'buk': buk})
    return pd.DataFrame(rows)

def calc_brier_multiclass(y_true, y_prob):
    from sklearn.metrics import brier_score_loss
    n_classes = y_prob.shape[1]
    brier_sum = 0
    for c in range(n_classes):
        y_binary = (y_true == c).astype(int)
        brier_sum += brier_score_loss(y_binary, y_prob[:, c])
    return brier_sum / n_classes

def calc_metrics(df):
    res = {}
    yt = df['actual'].values
    
    for name, col in [("LR", "lr"), ("Poisson", "poi"), ("Buk", "buk")]:
        yp = np.array(df[col].tolist())
        try:
            ll = log_loss(yt, yp, labels=[0, 1, 2])
        except ValueError:
            ll = float('nan')
        
        acc = (np.argmax(yp, axis=1) == yt).mean()
        
        yp_binary = np.zeros_like(yp)
        for i, label in enumerate(yt):
            yp_binary[i, label] = 1
        
        try:
            auc = roc_auc_score(yp_binary, yp, average='macro', multi_class='ovr')
        except ValueError:
            auc = float('nan')
        
        brier = calc_brier_multiclass(yt, yp)
        
        res[name] = {'logloss': ll, 'accuracy': acc, 'brier': brier, 'auc': auc}
    
    yp_random = np.array([[1/3, 1/3, 1/3]] * len(yt))
    try:
        ll_random = log_loss(yt, yp_random, labels=[0, 1, 2])
    except ValueError:
        ll_random = float('nan')
    acc_random = (np.argmax(yp_random, axis=1) == yt).mean()
    brier_random = calc_brier_multiclass(yt, yp_random)
    res['Random'] = {'logloss': ll_random, 'accuracy': acc_random, 'brier': brier_random, 'auc': 0.5}
    
    return res

def calc_simulation(df, model_col, odds_type, threshold=None):
    np.random.seed(42)
    bankroll = 100
    results = []
    for _, r in df.iterrows():
        probs = r[model_col]
        pred = np.argmax(probs)
        model_prob = probs[pred]
        
        odds_H = r['odds_H']
        odds_D = r['odds_D']
        odds_A = r['odds_A']
        
        implied_sum = (1/odds_H) + (1/odds_D) + (1/odds_A)
        margin = implied_sum - 1
        
        if odds_type == 'fair':
            bet_odds = odds_H * implied_sum if pred == 0 else (odds_D * implied_sum if pred == 1 else odds_A * implied_sum)
        else:
            bet_odds = odds_H if pred == 0 else (odds_D if pred == 1 else odds_A)
        
        ev = model_prob * bet_odds - 1
        
        if threshold is not None and threshold > 0 and ev < threshold:
            results.append({'bankroll': bankroll, 'bet': 0, 'return': 0, 'ev': ev, 'won': None})
            continue
        
        won = pred == r['actual']
        ret = bet_odds if won else 0
        bankroll += ret - 1
        results.append({'bankroll': bankroll, 'bet': 1, 'return': ret, 'ev': ev, 'won': won})
    return pd.DataFrame(results)

def load_builtin(league):
    import duckdb
    conn = duckdb.connect(DB_PATH)
    table_exists = conn.execute(f"SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'features_{league}'").fetchone()[0] > 0
    
    if table_exists:
        df = conn.execute(f"""
            SELECT m.match_date, m.home_team, m.away_team, m.season, m.result,
                   m.home_goals, m.away_goals,
                   m.avg_odds_H, m.avg_odds_D, m.avg_odds_A,
                   f.form_diff_5, f.form_diff_10, f.form_diff_15,
                   f.gd_diff_5, f.gd_diff_10, f.gd_diff_15,
                   f.home_advantage, f.home_away_form_diff, f.league_pos_diff
            FROM matches_raw m
            LEFT JOIN features_{league} f ON m.match_date = f.match_date
                AND m.home_team = f.home_team AND m.away_team = f.away_team
            WHERE m.league = '{league}'
            ORDER BY m.match_date
        """).df()
    else:
        df = conn.execute(f"""
            SELECT m.match_date, m.home_team, m.away_team, m.season, m.result,
                   m.home_goals, m.away_goals,
                   m.avg_odds_H, m.avg_odds_D, m.avg_odds_A
            FROM matches_raw m
            WHERE m.league = '{league}'
            ORDER BY m.match_date
        """).df()
    conn.close()
    return df

def validate(league, train_seasons, test_seasons):
    df = load_builtin(league)
    train = df[df['season'].isin(train_seasons)]
    model, scaler = train_lr(train)
    test = df[df['season'].isin(test_seasons)].dropna(subset=['avg_odds_H', 'form_diff_5'])
    preds = add_predictions(df, test, model, scaler)
    if preds.empty:
        return None
    return {'league': league, 'n_matches': len(preds), 'metrics': calc_metrics(preds),
            'train': train_seasons, 'test': test_seasons}

if __name__ == "__main__":
    import json
    import duckdb
    conn = duckdb.connect(DB_PATH)
    leagues = [r[0] for r in conn.execute("SELECT DISTINCT league FROM matches_raw ORDER BY league").fetchall()]
    conn.close()
    TRAIN = ["1617", "1718", "1819", "1920", "2021", "2122", "2223", "2324"]
    TEST = ["2425", "2526"]
    for league in leagues:
        r = validate(league, TRAIN, TEST)
        if r:
            print(json.dumps(r, indent=2))
