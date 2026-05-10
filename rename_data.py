import os
import pandas as pd

DATA_DIR = "data"

COL_MAP = {
    'match_date': ['Date', 'date', 'match_date'],
    'home_team': ['Home', 'HomeTeam', 'home', 'home_team'],
    'away_team': ['Away', 'AwayTeam', 'away', 'away_team'],
    'home_goals': ['HG', 'FTHG', 'home_goals'],
    'away_goals': ['AG', 'FTAG', 'away_goals'],
    'result': ['Res', 'FTR', 'result'],
    'season': ['Season', 'season'],
}

ODDS_SETS = [
    ('B365H', 'B365D', 'B365A'),
    ('PSCH', 'PSCD', 'PSCA'),
    ('AvgCH', 'AvgCD', 'AvgCA'),
    ('BFECH', 'BFECD', 'BFECA'),
    ('PSH', 'PSD', 'PSA'),
    ('WHH', 'WHD', 'WHA'),
    ('VCH', 'VCD', 'VCA'),
]

REQUIRED = ['match_date', 'home_team', 'away_team', 'home_goals', 'away_goals', 'result']

def read_csv(source):
    delimiters = [',', ';', '\t', '|']
    is_path = isinstance(source, str)
    for sep in delimiters:
        try:
            if is_path:
                test = pd.read_csv(source, sep=sep, nrows=5)
                if len(test.columns) > 1:
                    return pd.read_csv(source, sep=sep)
            else:
                source.seek(0)
                test = pd.read_csv(source, sep=sep, nrows=5)
                if len(test.columns) > 1:
                    source.seek(0)
                    return pd.read_csv(source, sep=sep)
        except Exception:
            if not is_path:
                source.seek(0)
    if is_path:
        return pd.read_csv(source)
    source.seek(0)
    return pd.read_csv(source)

def normalize(df):
    df.columns = df.columns.str.strip().str.replace('\ufeff', '', regex=False)
    cols = [c for c in df.columns]
    renamed = {}
    for std, aliases in COL_MAP.items():
        for a in aliases:
            if a in cols:
                renamed[a] = std
                cols.remove(a)
                break
    odds_src = None
    for h, d, a in ODDS_SETS:
        if h in cols and d in cols and a in cols:
            renamed.update({h: 'avg_odds_H', d: 'avg_odds_D', a: 'avg_odds_A'})
            odds_src = h.replace('H', '')
            cols.remove(h); cols.remove(d); cols.remove(a)
            break
    return df.copy().rename(columns=renamed), odds_src

def parse_dates(df):
    if 'match_date' not in df.columns:
        return df.copy()
    if pd.api.types.is_datetime64_any_dtype(df['match_date']):
        return df.copy()
    
    parsed = pd.to_datetime(df['match_date'], dayfirst=True, format='mixed', errors='coerce')
    df['match_date'] = parsed
    return df.copy()

def derive_season(dt):
    if pd.isna(dt):
        return "unknown"
    y = dt.year
    m = dt.month
    if m >= 7:
        return f"{str(y)[2:]}{str(y+1)[2:]}"
    else:
        return f"{str(y-1)[2:]}{str(y)[2:]}"

def normalize_season_value(s):
    if pd.isna(s):
        return None
    s = str(s).strip()
    if len(s) == 5 and '/' in s:
        parts = s.split('/')
        if len(parts) == 2:
            y1 = parts[0][-2:]
            y2 = parts[1][-2:]
            return f"{y1}{y2}"
    if len(s) == 4 and s.isdigit():
        return s
    return None

def ensure_season(df):
    needs_derive = False
    if 'season' not in df.columns:
        needs_derive = True
    else:
        normalized = df['season'].apply(normalize_season_value)
        if normalized.isna().any() or (normalized == 'unknown').any():
            needs_derive = True
    
    if needs_derive:
        if 'season' in df.columns:
            del df['season']
        if 'match_date' in df.columns:
            df['season'] = df['match_date'].apply(derive_season)
    return df.copy()

def load_and_normalize(source):
    df = read_csv(source)
    df, odds_src = normalize(df)
    df = parse_dates(df)
    df = ensure_season(df)
    if 'match_date' in df.columns:
        df = df.sort_values('match_date').reset_index(drop=True)
    return df.copy(), odds_src

def get_season_from_file(filepath):
    df, _ = load_and_normalize(filepath)
    if 'match_date' not in df.columns or df['match_date'].isna().all():
        return "unknown"
    first_date = df['match_date'].dropna().iloc[0]
    return derive_season(first_date)

def rename_csv(base_dir):
    for league in sorted(os.listdir(base_dir)):
        path = os.path.join(base_dir, league)
        if not os.path.isdir(path):
            continue
        for fname in sorted(os.listdir(path)):
            if not fname.lower().endswith('.csv'):
                continue
            fpath = os.path.join(path, fname)
            
            base = os.path.splitext(fname)[0]
            if '_' in base and len(base.split('_')[-1]) == 4:
                parts = base.split('_')
                if parts[-1].isdigit():
                    continue
            
            try:
                season = get_season_from_file(fpath)
                new_name = f"{league}_{season}.csv"
                new_path = os.path.join(path, new_name)
                if fpath != new_path:
                    os.rename(fpath, new_path)
                    print(f"  {fname} -> {new_name}")
            except Exception as e:
                print(f"  {fname}: {e}")

if __name__ == "__main__":
    print("Renaming CSV files...")
    rename_csv(DATA_DIR)
    print("Done")