import os
import duckdb
import pandas as pd
import numpy as np

DB_PATH = os.path.join("database", "football.db")
LEAGUES = ["PL", "SA", "LL"]

def get_connection():
    return duckdb.connect(DB_PATH)

def add_rolling_features(df: pd.DataFrame, n_values: list = [5, 10, 15]) -> pd.DataFrame:
    df = df.sort_values("match_date").reset_index(drop=True)
    team_history = {}

    for n in n_values:
        df[f"home_form_{n}"] = 0.0
        df[f"away_form_{n}"] = 0.0
        df[f"form_diff_{n}"] = 0.0
        df[f"home_gd_{n}"] = 0.0
        df[f"away_gd_{n}"] = 0.0
        df[f"gd_diff_{n}"] = 0.0

    home_advantage = {
        "PL": 0.464,
        "SA": 0.461,
        "LL": 0.478
    }

    for idx, row in df.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        league = row["league"]

        for team in [home, away]:
            if team not in team_history:
                team_history[team] = {"points": [], "goal_diff": []}

        for n in n_values:
            if len(team_history[home]["points"]) >= n:
                home_form = sum(team_history[home]["points"][-n:])
            else:
                home_form = 0
            if len(team_history[away]["points"]) >= n:
                away_form = sum(team_history[away]["points"][-n:])
            else:
                away_form = 0

            df.at[idx, f"home_form_{n}"] = home_form
            df.at[idx, f"away_form_{n}"] = away_form
            df.at[idx, f"form_diff_{n}"] = home_form - away_form

            if len(team_history[home]["goal_diff"]) >= n:
                home_gd = sum(team_history[home]["goal_diff"][-n:])
            else:
                home_gd = 0
            if len(team_history[away]["goal_diff"]) >= n:
                away_gd = sum(team_history[away]["goal_diff"][-n:])
            else:
                away_gd = 0

            df.at[idx, f"home_gd_{n}"] = home_gd
            df.at[idx, f"away_gd_{n}"] = away_gd
            df.at[idx, f"gd_diff_{n}"] = home_gd - away_gd

        df.at[idx, "home_advantage"] = home_advantage.get(league, 0.46)

        if row["result"] == "H":
            team_history[home]["points"].append(3)
            team_history[away]["points"].append(0)
            team_history[home]["goal_diff"].append(row["home_goals"] - row["away_goals"])
            team_history[away]["goal_diff"].append(row["away_goals"] - row["home_goals"])
        elif row["result"] == "A":
            team_history[home]["points"].append(0)
            team_history[away]["points"].append(3)
            team_history[home]["goal_diff"].append(row["home_goals"] - row["away_goals"])
            team_history[away]["goal_diff"].append(row["away_goals"] - row["home_goals"])
        else:
            team_history[home]["points"].append(1)
            team_history[away]["points"].append(1)
            team_history[home]["goal_diff"].append(row["home_goals"] - row["away_goals"])
            team_history[away]["goal_diff"].append(row["away_goals"] - row["home_goals"])

    return df

def build_features_tables():
    conn = get_connection()
    n_values = [5, 10, 15]

    for league in LEAGUES:
        print(f"Processing {league}...")
        query = f"""
            SELECT match_date, home_team, away_team, home_goals, away_goals, 
                   result, season, league,
                   avg_odds_H, avg_odds_D, avg_odds_A,
                   implied_H, implied_D, implied_A
            FROM matches_raw
            WHERE league = '{league}'
            ORDER BY match_date
        """
        df = conn.execute(query).df()
        df = add_rolling_features(df, n_values)

        league_cols = pd.get_dummies(df["league"], prefix="league")
        df = pd.concat([df, league_cols], axis=1)

        feature_cols = [
            "match_date", "home_team", "away_team", "result", "season", "league",
            "avg_odds_H", "avg_odds_D", "avg_odds_A",
            "implied_H", "implied_D", "implied_A"
        ]
        feature_cols += [f"form_diff_{n}" for n in n_values]
        feature_cols += [f"gd_diff_{n}" for n in n_values]
        feature_cols += ["home_advantage"]
        feature_cols += [c for c in df.columns if c.startswith("league_")]

        df_features = df[feature_cols].copy()

        conn.execute(f"DROP TABLE IF EXISTS features_{league}")
        conn.register(f"features_{league}", df_features)
        conn.execute(f"CREATE TABLE features_{league} AS SELECT * FROM df_features")
        conn.unregister(f"features_{league}")
        print(f"  Created features_{league} with {len(df_features)} rows")

    conn.close()

if __name__ == "__main__":
    build_features_tables()