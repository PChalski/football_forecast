import os
import duckdb
import pandas as pd
import math

DB_PATH = os.path.join("database", "football.db")
LEAGUES = ["PL", "SA", "LL"]

# Elo rating built according to https://www.eloratings.net/about
K_BASE = 30
HOME_ADVANTAGE = 100
START_RATING = 1500

def get_connection():
    return duckdb.connect(DB_PATH)

def k_adjustment(goal_diff):
    if goal_diff >= 4:
        return K_BASE + K_BASE * (0.75 + (goal_diff - 3) / 8)
    elif goal_diff == 3:
        return K_BASE + K_BASE * 0.75
    elif goal_diff == 2:
        return K_BASE + K_BASE * 0.5
    return K_BASE

def expected_result(rating_diff):
    return 1 / (math.pow(10, -rating_diff / 400) + 1)

def compute_elo_ratings():
    conn = get_connection()
    conn.execute("DROP TABLE IF EXISTS elo_ratings")

    all_elo = []

    for league in LEAGUES:
        print(f"Processing {league}...")

        query = f"""
            SELECT match_date, home_team, away_team, home_goals, away_goals, result, season
            FROM matches_raw
            WHERE league = '{league}'
            ORDER BY match_date
        """
        df = conn.execute(query).df()

        ratings = {}

        for idx, row in df.iterrows():
            home = row["home_team"]
            away = row["away_team"]

            if home not in ratings:
                ratings[home] = START_RATING
            if away not in ratings:
                ratings[away] = START_RATING

            rating_home_pre = ratings[home]
            rating_away_pre = ratings[away]

            goal_diff = abs(row["home_goals"] - row["away_goals"])

            if row["result"] == "H":
                actual_home = 1
                actual_away = 0
            elif row["result"] == "A":
                actual_home = 0
                actual_away = 1
            else:
                actual_home = 0.5
                actual_away = 0.5

            dr = (rating_home_pre + HOME_ADVANTAGE) - rating_away_pre
            we_home = expected_result(dr)
            we_away = 1 - we_home

            k = k_adjustment(goal_diff)

            ratings[home] = rating_home_pre + k * (actual_home - we_home)
            ratings[away] = rating_away_pre + k * (actual_away - we_away)

            all_elo.append({
                "match_date": row["match_date"],
                "home_team": home,
                "away_team": away,
                "league": league,
                "elo_home_pre": rating_home_pre,
                "elo_away_pre": rating_away_pre,
                "elo_diff": (rating_home_pre + HOME_ADVANTAGE) - rating_away_pre,
                "result": row["result"]
            })

    df_elo = pd.DataFrame(all_elo)

    df_elo["elo_prob_H_d"] = df_elo["elo_diff"].apply(
        lambda d: 1 / (1 + math.pow(10, -d / 400))
    )
    df_elo["elo_prob_A_d"] = df_elo["elo_diff"].apply(
        lambda d: 1 / (1 + math.pow(10, d / 400))
    )
    df_elo["draw_factor"] = 0.26
    
    df_elo["elo_prob_H"] = df_elo["elo_prob_H_d"] * (1 - df_elo["draw_factor"])
    df_elo["elo_prob_A"] = df_elo["elo_prob_A_d"] * (1 - df_elo["draw_factor"])
    df_elo["elo_prob_D"] = df_elo["draw_factor"]

    conn.register("df_elo", df_elo)
    conn.execute("""
        CREATE TABLE elo_ratings AS
        SELECT * FROM df_elo
        ORDER BY match_date
    """)
    conn.unregister("df_elo")

    print(f"Created elo_ratings with {len(df_elo)} rows")
    conn.close()

if __name__ == "__main__":
    compute_elo_ratings()