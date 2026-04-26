import os
import pandas as pd

DATA_DIR = "data"
DATE_COL = "Date" 

def extract_season(df : pd.DataFrame) -> str:
    """
    Returns season years as a string ('1617' for 2016/2017) after checking year of first match of the season.
    """
    first_date_str = df[DATE_COL].dropna().iloc[0] # Dropping NaN values, finding date of first game

    first_date = pd.to_datetime(first_date_str, dayfirst=True) # date uses DD/MM/YYYY format
    start_year = first_date.year
   
    end_year = start_year + 1
    
    return f"{str(start_year)[-2:]}{str(end_year)[-2:]}" # last 2 digits of each year

def rename_csv(base_dir):
    """
    Renames csv files from base_dir to its {league}_{season}
    """
    
    for league_name in os.listdir(base_dir): # csv files for each league in different directory "league_name"
        league_path = os.path.join(base_dir, league_name)
        
        if not os.path.isdir(league_path):
            continue

        for file in os.listdir(league_path):
            if not file.lower().endswith(".csv"): # renaming only files with data (.csv)
                continue

            file_path = os.path.join(league_path, file)
            try:
                df = pd.read_csv(file_path)
                if DATE_COL not in df.columns:
                    print(f" Column {DATE_COL} not in {file_path}, skipping...")
                    continue

                season = extract_season(df)

                new_name = f"{league_name}_{season}.csv"
                new_path = os.path.join(league_path, new_name)


                os.rename(file_path, new_path)
                print(f"Renamed {file} → {new_name}")
            except Exception as e:
                print(f"Error: {file_path}: {e}")

if __name__ == "__main__":
    rename_csv(DATA_DIR)
    print("Done")