import os
import duckdb
import pandas as pd
from load_data import load_league

DB_PATH = os.path.join("database", "results.db")

LEAGUES = {"PL": "Premier League", "SA": "Serie A", "LL": "La Liga"}

def create_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    return duckdb.connect(DB_PATH)

def add_avg_odds(df):
    df["avg_odds_H"] = df["B365H"]
    df["avg_odds_D"] = df["B365D"]
    df["avg_odds_A"] = df["B365A"]
    return df

def build_raw_table():
    conn = create_connection()
    conn.execute("DROP TABLE IF EXISTS matches_raw")
    all_matches = []
    for league_code, league_name in LEAGUES.items():
        df = load_league(league_code)
        df["league"] = league_code
        df = add_avg_odds(df)
        all_matches.append(df)
    full = pd.concat(all_matches, ignore_index=True)
    full = full.sort_values("Date").reset_index(drop=True)
    col_map = {"Date": "match_date", "HomeTeam": "home_team", "AwayTeam": "away_team",
               "FTHG": "home_goals", "FTAG": "away_goals", "FTR": "result", "Season": "season"}
    full = full.rename(columns=col_map)
    conn.register("full_df", full)
    conn.execute("CREATE TABLE matches_raw AS SELECT * FROM full_df ORDER BY match_date")
    conn.unregister("full_df")
    print(f"Created matches_raw with {conn.execute('SELECT COUNT(*) FROM matches_raw').fetchone()[0]} rows")
    conn.close()

if __name__ == "__main__":
    build_raw_table()