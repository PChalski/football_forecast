# src/create_db.py
import os
import duckdb
import pandas as pd
from load_data import load_league

DB_PATH = os.path.join("database", "results.db")
DATA_DIR = "data"

LEAGUES = {
    "PL": "Premier League",
    "SA": "Serie A",
    "LL": "La Liga"
}

ODDS_COLS_H = ["B365H", "PSH", "WHH", "VCH"]
ODDS_COLS_D = ["B365D", "PSD", "WHD", "VCD"]
ODDS_COLS_A = ["B365A", "PSA", "WHA", "VCA"]

def create_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    return duckdb.connect(DB_PATH)

def add_avg_odds(df: pd.DataFrame) -> pd.DataFrame:
    df_odds_H = df[ODDS_COLS_H]
    df_odds_D = df[ODDS_COLS_D]
    df_odds_A = df[ODDS_COLS_A]
    df["avg_odds_H"] = df_odds_H.mean(axis=1, skipna=True)
    df["avg_odds_D"] = df_odds_D.mean(axis=1, skipna=True)
    df["avg_odds_A"] = df_odds_A.mean(axis=1, skipna=True)
    return df

def implied_probs(row):
    odds = [row["avg_odds_H"], row["avg_odds_D"], row["avg_odds_A"]]
    if any(pd.isna(odds)) or any(o == 0 for o in odds):
        return pd.Series([pd.NA, pd.NA, pd.NA])
    implied = [1/o for o in odds]
    total = sum(implied)
    overround = total - 1
    margin = overround / 3
    probs = [(1/o - margin) for o in odds]
    total_prob = sum(probs)
    probs = [p/total_prob for p in probs]
    return pd.Series(probs, index=["implied_H", "implied_D", "implied_A"])

def build_raw_table():
    conn = create_connection()
    conn.execute("DROP TABLE IF EXISTS matches_raw")

    all_matches = []
    for league_code, league_name in LEAGUES.items():
        df = load_league(league_code)
        df["league"] = league_code
        df = add_avg_odds(df)
        probs = df.apply(implied_probs, axis=1)
        df["implied_H"] = probs["implied_H"]
        df["implied_D"] = probs["implied_D"]
        df["implied_A"] = probs["implied_A"]
        all_matches.append(df)

    full = pd.concat(all_matches, ignore_index=True)
    full = full.sort_values("Date").reset_index(drop=True)

    col_map = {
        "Date": "match_date",
        "HomeTeam": "home_team",
        "AwayTeam": "away_team",
        "FTHG": "home_goals",
        "FTAG": "away_goals",
        "FTR": "result",
        "Season": "season"
    }
    full = full.rename(columns=col_map)

    conn.register("full_df", full)
    conn.execute("""
        CREATE TABLE matches_raw AS
        SELECT * FROM full_df
        ORDER BY match_date
    """)
    conn.unregister("full_df")
    print(f"Created matches_raw with {conn.execute('SELECT COUNT(*) FROM matches_raw').fetchone()[0]} rows")
    conn.close()

if __name__ == "__main__":
    build_raw_table()