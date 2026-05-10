import os
import pandas as pd
from rename_data import load_and_normalize

DATA_DIR = "data"

def load_league(league_name: str) -> pd.DataFrame:
    league_path = os.path.join(DATA_DIR, league_name)
    if not os.path.isdir(league_path):
        raise FileNotFoundError(f"Directory not found: {league_path}")

    csv_files = sorted(f for f in os.listdir(league_path) if f.lower().endswith(".csv"))
    if not csv_files:
        raise FileNotFoundError(f"No csv files in {league_path}")

    dfs = []
    for fname in csv_files:
        file_path = os.path.join(league_path, fname)
        df, _ = load_and_normalize(file_path)
        dfs.append(df)

    full = pd.concat(dfs, ignore_index=True).sort_values("match_date").reset_index(drop=True)
    return full


if __name__ == "__main__":
    for league in ["PL", "SA", "LL", "L1"]:
        try:
            df = load_league(league)
            print(f"{league}: {len(df)} matches, {df['season'].nunique()} seasons, "
                  f"{df['match_date'].min().date()} - {df['match_date'].max().date()}")
        except FileNotFoundError as e:
            print(f"{league}: {e}")