import duckdb
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

FEATURE_COLS = ['form_diff_5', 'form_diff_10', 'form_diff_15',
              'gd_diff_5', 'gd_diff_10', 'gd_diff_15',
              'home_advantage', 'home_away_form_diff', 'league_pos_diff']

def train(league, train_seasons):
    conn = duckdb.connect("database/results.db")
    train_str = ",".join([f"'{s}'" for s in train_seasons])
    
    df = conn.execute(f"""
        SELECT result, {', '.join(FEATURE_COLS)}
        FROM features_{league}
        WHERE season IN ({train_str})
    """).df()
    conn.close()
    
    df = df.dropna(subset=['form_diff_5', 'gd_diff_5'])
    
    X = df[FEATURE_COLS].values
    y = df['result'].map({'H': 0, 'D': 1, 'A': 2}).values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = LogisticRegression(solver='lbfgs', max_iter=2000, C=0.5)
    model.fit(X_scaled, y)
    
    return model, scaler

def predict(model, scaler, features_dict):
    X = np.array([[features_dict.get(f, 0) for f in FEATURE_COLS]])
    X_scaled = scaler.transform(X)
    
    proba = model.predict_proba(X_scaled)[0]
    
    return {"H": proba[0], "D": proba[1], "A": proba[2]}

if __name__ == "__main__":
    model, scaler = train("PL", ["1617", "1718", "1819", "1920", "2021", "2122", "2223", "2324"])
    
    test_features = {
        "form_diff_5": 9, "form_diff_10": 15, "form_diff_15": 18,
        "gd_diff_5": 5, "gd_diff_10": 10, "gd_diff_15": 12,
        "home_advantage": 0.464, "home_away_form_diff": 3, "league_pos_diff": -2
    }
    
    probs = predict(model, scaler, test_features)
    print(f"PL: H={probs['H']:.1%}, D={probs['D']:.1%}, A={probs['A']:.1%}")