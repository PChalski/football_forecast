import os
import duckdb
import pandas as pd
from load_data import load_league
from build_features import add_rolling_features

DB_PATH = os.path.join("database", "results.db")

def get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    return duckdb.connect(DB_PATH)

def discover_leagues():
    leagues = []
    for d in sorted(os.listdir("data")):
        path = os.path.join("data", d)
        if os.path.isdir(path) and any(f.lower().endswith(".csv") for f in os.listdir(path)):
            leagues.append(d)
    return leagues

def build_raw_table():
    conn = get_connection()
    conn.execute("DROP TABLE IF EXISTS matches_raw")
    leagues = discover_leagues()
    all_matches = []
    for league_code in leagues:
        try:
            df = load_league(league_code)
            df["league"] = league_code
            all_matches.append(df)
        except FileNotFoundError:
            pass
    if not all_matches:
        conn.close()
        return
    full = pd.concat(all_matches, ignore_index=True)
    full = full.sort_values("match_date").reset_index(drop=True)
    conn.register("full_df", full)
    conn.execute("CREATE TABLE matches_raw AS SELECT * FROM full_df ORDER BY match_date")
    conn.unregister("full_df")
    conn.close()

def build_features_tables():
    conn = get_connection()
    leagues = [r[0] for r in conn.execute("SELECT DISTINCT league FROM matches_raw").fetchall()]
    for league in leagues:
        table_name = f"features_{league}"
        conn.execute(f"DROP TABLE IF EXISTS {table_name}")
        df = conn.execute(f"SELECT * FROM matches_raw WHERE league = '{league}' ORDER BY match_date").df()
        df = add_rolling_features(df).copy()
        conn.register("feat_df", df)
        conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM feat_df ORDER BY match_date")
        conn.unregister("feat_df")
    conn.close()

if __name__ == "__main__":
    build_raw_table()
    build_features_tables()
