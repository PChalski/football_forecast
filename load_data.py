import os
import pandas as pd
import numpy as np

DATA_DIR = "data"
DATE_COL = "Date"
def load_league(league_name: str) -> pd.DataFrame:
    """
    Loads all csv files from selected league into one dataframe

    """
    league_path = os.path.join(DATA_DIR, league_name)

    if not os.path.isdir(league_path):
        raise FileNotFoundError(f"Directory not found: {league_path}")

    # all csv files in dir
    csv_files = [
        f for f in os.listdir(league_path)
        if f.lower().endswith(".csv")
    ]

    if not csv_files:
        raise FileNotFoundError(f"No csv files in {league_path}")

    csv_files.sort()

    dfs = []
    seasons = []
    for fname in csv_files:
        file_path = os.path.join(league_path, fname)
        df = pd.read_csv(file_path)

        # "date" conversion
        if DATE_COL in df.columns:
            df[DATE_COL] = pd.to_datetime(df[DATE_COL], format="mixed", dayfirst=True)

        season = fname[3:7] # assuming 2 signs for league name and _ before season years
        seasons.append(season)
        
        dfs.append(df)

    full_df = pd.concat(dfs, ignore_index=True).copy()
    full_df = full_df.sort_values(DATE_COL).reset_index(drop=True)

    season_list = np.concatenate([
    np.repeat(season, len(df_i)) for season, df_i in zip(seasons, dfs)
    ])
    
    full_df["Season"] = season_list
    
    return full_df


if __name__ == "__main__":

    df = load_league("PL")
    print(f"Number of matches: {len(df)}")
    print(f"Number of seasons: {df['Season'].nunique()}")
    print(f"Time range {df['Date'].min().date()} - {df['Date'].max().date()}")
    print(df.head())