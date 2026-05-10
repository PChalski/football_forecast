import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
from validation import train_lr, add_predictions, calc_metrics, load_builtin, FEATURE_COLS, predict_poisson
from build_features import add_rolling_features
from rename_data import load_and_normalize, REQUIRED
from betting_simulation import get_summary

def format_season(s):
    if len(str(s)) >= 4:
        return f"{str(s)[:2]}/{str(s)[2:]}"
    return str(s)

def format_league(lg):
    names = {'PL': 'Premier League', 'LL': 'La Liga', 'SA': 'Serie A', 'POL': 'Ekstraklasa'}
    return names.get(lg, lg)

import os
if not os.path.exists("database/results.db"):
    st.warning("Creating database...")
    from create_db import build_raw_table, build_features_tables
    build_raw_table()
    build_features_tables()
    st.success("Database created!")

st.set_page_config(layout="wide")
st.title("Football Forecast")

for k in ["df", "preds", "model", "scaler", "train_seasons", "test_seasons", "league"]:
    st.session_state.setdefault(k, None)

if 'available_leagues' not in st.session_state:
    import duckdb
    conn = duckdb.connect("database/results.db")
    st.session_state.available_leagues = [r[0] for r in conn.execute("SELECT DISTINCT league FROM matches_raw ORDER BY league").fetchall()]
    st.session_state.leagues_with_features = [r[0].replace('features_', '') for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'features_%'").fetchall()]
    conn.close()

if 'trained_models' not in st.session_state:
    st.session_state.trained_models = {}

if 'all_training_results' not in st.session_state:
    st.session_state.all_training_results = []

if 'upload_counter' not in st.session_state:
    st.session_state.upload_counter = 0

tabs = st.tabs(["Training", "Insights", "Betting Simulation", "Predict"])

with tabs[0]:
    st.subheader("Train Model")
    
    src = st.radio("Source", ["Built-in", "Upload CSV"], horizontal=True, key="src_radio")
    
    if src == "Built-in":
        league = st.selectbox("League", st.session_state.available_leagues, format_func=format_league, key="train_league")
        
        if st.button("Load data", key="load_data_btn"):
            with st.spinner("Loading..."):
                df = load_builtin(league)
                if df.empty:
                    st.error(f"No data for {league}")
                else:
                    st.session_state.df = df
                    st.session_state.league = league
                    seasons = sorted(df['season'].unique())
                    st.session_state.all_seasons = seasons
                    st.success(f"{len(df)} matches - {len(seasons)} seasons")
    else:
        files = st.file_uploader("Upload CSV", type="csv", accept_multiple_files=True, key=f"upl_{st.session_state.upload_counter}")
        league_name = st.text_input("League name", key=f"league_name_{st.session_state.upload_counter}")
        
        if files and league_name and st.button("Load CSV", key="load_csv_btn"):
            with st.spinner("Loading..."):
                dfs, odds_info = [], set()
                for f in files:
                    df, odds_src = load_and_normalize(f)
                    dfs.append(df)
                    odds_info.add(odds_src)
                df = pd.concat(dfs, ignore_index=True).sort_values('match_date').reset_index(drop=True).copy()
                missing = [c for c in REQUIRED if c not in df.columns]
                if missing:
                    st.error(f"Missing: {missing}")
                else:
                    if any(c not in df.columns for c in FEATURE_COLS):
                        df['league'] = league_name
                        df = add_rolling_features(df).copy()
                    
                    import duckdb
                    conn = duckdb.connect("database/results.db")
                    features_table = f"features_{league_name}"
                    conn.execute(f"DROP TABLE IF EXISTS {features_table}")
                    conn.execute(f"CREATE TABLE {features_table} AS SELECT * FROM df")
                    matches_table = f"matches_raw"
                    existing = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='matches_raw'").fetchall()
                    if not existing:
                        conn.execute(f"CREATE TABLE {matches_table} AS SELECT * FROM df")
                    conn.close()
                    
                    st.session_state.df = df
                    st.session_state.league = league_name
                    seasons = sorted(df['season'].unique())
                    st.session_state.all_seasons = seasons
                    st.session_state.upload_counter += 1
                    st.success(f"{len(df)} matches - {len(seasons)} seasons")
                    st.rerun()
    
    if st.session_state.get('df') is not None:
        df = st.session_state.df
        league = st.session_state.league
        seasons = st.session_state.get('all_seasons', sorted(df['season'].unique()))
        
        if len(seasons) >= 2:
            default_train = seasons[-10:-2] if len(seasons) >= 10 else seasons[:-2]
            default_test = seasons[-2:]
            
            col1, col2 = st.columns(2)
            with col1:
                train_seasons = st.multiselect("Training seasons", options=seasons, default=default_train, key="train_seas")
            with col2:
                test_seasons = st.multiselect("Testing seasons", options=seasons, default=default_test, key="test_seas")
            
            if train_seasons and test_seasons:
                if set(train_seasons) & set(test_seasons):
                    st.error("Seasons cannot overlap")
                else:
                    train_df = df[df['season'].isin(train_seasons)].dropna(subset=FEATURE_COLS)
                    test_df = df[df['season'].isin(test_seasons)].dropna(subset=['avg_odds_H'] + FEATURE_COLS)
                    
                    st.info(f"Train: {len(train_df)} matches | Test: {len(test_df)} matches")
                    
                    if st.button("Predict", type="primary", key="predict_btn"):
                        with st.spinner("Training..."):
                            model, scaler = train_lr(train_df)
                            preds = add_predictions(df, test_df, model, scaler)
                            metrics = calc_metrics(preds)
                            
                            result = {
                                'league': league,
                                'n_matches': len(preds),
                                'train_seasons': train_seasons,
                                'test_seasons': test_seasons,
                                'metrics': metrics,
                                'preds': preds,
                                'model': model,
                                'scaler': scaler
                            }
                            
                            st.session_state.trained_models[league] = {
                                'model': model,
                                'scaler': scaler,
                                'train_seasons': train_seasons,
                                'test_seasons': test_seasons
                            }
                            
                            existing_idx = next((i for i, r in enumerate(st.session_state.all_training_results) if r['league'] == league), None)
                            if existing_idx is not None:
                                st.session_state.all_training_results[existing_idx] = result
                            else:
                                st.session_state.all_training_results.append(result)
                            
                            st.session_state.df = None
                            st.rerun()
    
    st.divider()
    st.subheader("Trained Models")
    if st.session_state.trained_models:
        for lg, info in st.session_state.trained_models.items():
            train_str = f"{format_season(info['train_seasons'][0])}-{format_season(info['train_seasons'][-1])}"
            test_str = f"{format_season(info['test_seasons'][0])}-{format_season(info['test_seasons'][-1])}"
            st.write(f"**{format_league(lg)}**: Train {train_str}, Test {test_str}")
    else:
        st.info("No models trained yet")

with tabs[1]:
    if not st.session_state.get('all_training_results'):
        st.info("Train models first in Training tab")
    else:
        for r in st.session_state.all_training_results:
            st.subheader(f"{format_league(r['league'])} - {r['n_matches']} test matches")
            train_str = f"{format_season(r['train_seasons'][0])}-{format_season(r['train_seasons'][-1])}"
            test_str = f"{format_season(r['test_seasons'][0])}-{format_season(r['test_seasons'][-1])}"
            st.caption(f"Train: {train_str} | Test: {test_str}")
            
            df_m = pd.DataFrame(r['metrics']).T
            df_m.index = ['Logistic Regression', 'Poisson', 'Bookmaker', 'Random']
            df_m['logloss'] = df_m['logloss'].map("{:.4f}".format)
            df_m['brier'] = df_m['brier'].map("{:.4f}".format)
            df_m['auc'] = df_m['auc'].map("{:.3f}".format)
            df_m['accuracy'] = df_m['accuracy'].map("{:.1%}".format)
            st.dataframe(df_m, width='stretch')
            st.divider()

with tabs[2]:
    if not st.session_state.get('all_training_results'):
        st.info("Train models first")
    else:
        league_options = [r['league'] for r in st.session_state.all_training_results]
        league_display = {lg: format_league(lg) for lg in league_options}
        
        col1, col2 = st.columns(2)
        with col1:
            selected_league = st.selectbox("League", league_options, format_func=lambda x: league_display[x], key="bet_league")
        with col2:
            model = st.selectbox("Model (prediction)", ["lr", "poi", "Random"], format_func=lambda x: {"lr": "Logistic Regression", "poi": "Poisson", "Random": "Random"}[x], key="bet_model",
                                 help="Bookmaker odds are used as the betting source")
        
        result = next(r for r in st.session_state.all_training_results if r['league'] == selected_league)
        preds = result['preds']
        
        col_odds, col_ev = st.columns(2)
        with col_odds:
            odds_type = st.selectbox("Odds type for comparison", ["fair", "margined"], key="bet_odds", 
                                      help="Fair = model odds without margin, Margined = with margin")
        with col_ev:
            no_limit = st.checkbox("No EV limit (bet all)", key="no_limit")
            if no_limit:
                min_ev = -1.0
            else:
                min_ev = st.slider("Min EV threshold", -0.2, 0.5, 0.0, 0.01, key="bet_ev", 
                                   help="Only bet if EV > threshold")
        
        np.random.seed(42)
        all_margins = []
        sim_results = []
        for _, r in preds.iterrows():
            odds_H = r['odds_H']
            odds_D = r['odds_D']
            odds_A = r['odds_A']
            implied_sum = (1/odds_H) + (1/odds_D) + (1/odds_A)
            margin = implied_sum - 1
            all_margins.append(margin)
            
            odds_fair = [odds_H * implied_sum, odds_D * implied_sum, odds_A * implied_sum]
            odds_margined = [odds_H, odds_D, odds_A]
            
            if model == 'Random':
                probs = [1/3, 1/3, 1/3]
                best_idx = np.random.choice([0, 1, 2])
                ev_fair = [probs[0] * odds_fair[0] - 1, probs[1] * odds_fair[1] - 1, probs[2] * odds_fair[2] - 1]
                ev_margined = [probs[0] * odds_margined[0] - 1, probs[1] * odds_margined[1] - 1, probs[2] * odds_margined[2] - 1]
            else:
                probs = r[model]
                ev_fair = [probs[0] * odds_fair[0] - 1, probs[1] * odds_fair[1] - 1, probs[2] * odds_fair[2] - 1]
                ev_margined = [probs[0] * odds_margined[0] - 1, probs[1] * odds_margined[1] - 1, probs[2] * odds_margined[2] - 1]
                
                if odds_type == 'fair':
                    evs = ev_fair
                else:
                    evs = ev_margined
                
                best_idx = np.argmax(evs)
            
            best_ev = evs[best_idx] if model != 'Random' else ev_fair[best_idx]
            best_odds = odds_fair[best_idx] if odds_type == 'fair' else odds_margined[best_idx]
            
            outcome_map = {0: 'H', 1: 'D', 2: 'A'}
            actual_map = {0: 'H', 1: 'D', 2: 'A'}
            
            sim_results.append({
                'date': r.get('match_date', r.get('date', '')),
                'home': r['home'],
                'away': r['away'],
                'actual': actual_map.get(r['actual'], r['actual']),
                'pred': outcome_map[best_idx],
                'prob_H': f"{probs[0]:.1%}",
                'prob_D': f"{probs[1]:.1%}",
                'prob_A': f"{probs[2]:.1%}",
                'odds_H': odds_H,
                'odds_D': odds_D,
                'odds_A': odds_A,
                'odds_fair_H': f"{odds_fair[0]:.2f}",
                'odds_fair_D': f"{odds_fair[1]:.2f}",
                'odds_fair_A': f"{odds_fair[2]:.2f}",
                'ev_fair': round(ev_fair[best_idx], 3),
                'ev_margined': round(ev_margined[best_idx], 3),
                'ev': best_ev,
                'best_idx': best_idx,
                'bet_odds': best_odds
            })
        
        sim_df = pd.DataFrame(sim_results)
        
        if odds_type == 'margined':
            st.caption(f"Avg margin: {np.mean(all_margins)*100:.2f}%")
        
        bankroll = 100
        bet_results = []
        for i, r in sim_df.iterrows():
            if r['ev'] >= min_ev:
                won = r['pred'] == r['actual']
                ret = r['bet_odds'] if won else 0
                bankroll += ret - 1
                bet_results.append({
                    **r.to_dict(),
                    'bet': 1,
                    'won': won,
                    'return': ret,
                    'bankroll': bankroll
                })
            else:
                bet_results.append({
                    **r.to_dict(),
                    'bet': 0,
                    'won': None,
                    'return': 0,
                    'bankroll': bankroll
                })
        
        bet_df = pd.DataFrame(bet_results)
        bets = bet_df[bet_df['bet'] == 1]
        
        if len(bets) > 0:
            total_return = bets['return'].sum()
            n_bets = len(bets)
            profit = total_return - n_bets
            roi = profit / n_bets * 100
            hit_rate = bets['won'].mean() * 100
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Bets", n_bets)
            c2.metric("Starting bankroll", "1000")
            c3.metric("Profit", f"{profit:.1f}")
            c4.metric("Hit rate", f"{hit_rate:.1f}%")
            
            st.line_chart(bet_df.set_index(bet_df.index)[['bankroll']], height=300)
            
            st.subheader("Bets")
            display_cols = ['date', 'home', 'away', 'pred', 'actual', 'odds_H', 'odds_D', 'odds_A', 
                            'odds_fair_H', 'odds_fair_D', 'odds_fair_A', 'ev_fair', 'ev_margined', 'won', 'return']
            st.dataframe(bets[display_cols].head(50), width='stretch')
        else:
            st.warning("No bets with selected criteria")

with tabs[3]:
    st.subheader("Predict Match")
    
    if not st.session_state.trained_models:
        st.info("Train a model first in Training tab")
    else:
        col1, col2 = st.columns(2)
        with col1:
            predict_league_keys = list(st.session_state.trained_models.keys())
            predict_league_display = {k: format_league(k) for k in predict_league_keys}
            predict_league = st.selectbox("League", predict_league_keys, format_func=lambda x: predict_league_display[x], key="pred_league")
        with col2:
            predict_model = st.selectbox("Model", ["LR", "Poisson"], key="pred_model")
        
        model_info = st.session_state.trained_models[predict_league]
        
        result = next((r for r in st.session_state.all_training_results if r['league'] == predict_league), None)
        
        session_df = st.session_state.get('df')
        session_league = st.session_state.get('league')
        
        if result and 'preds' in result:
            df = result['preds']
            use_builtin_for_poisson = True
        elif session_league == predict_league and session_df is not None:
            df = session_df
            use_builtin_for_poisson = False
        else:
            df = load_builtin(predict_league)
            use_builtin_for_poisson = False
        has_features = session_df is not None and all(c in session_df.columns for c in FEATURE_COLS)
        
        teams = sorted(set(df['home'].unique()) | set(df['away'].unique())) if 'home' in df.columns else sorted(set(df['home_team'].unique()) | set(df['away_team'].unique()))
        
        col1, col2 = st.columns(2)
        with col1:
            home_team = st.selectbox("Home team", teams, key="home_team")
        with col2:
            away_team = st.selectbox("Away team", teams, key="away_team")
        
        if home_team == away_team:
            st.error("Teams must be different")
        else:
            feat = None
            
            if has_features:
                match = session_df[(session_df['home_team'] == home_team) | (session_df['home'] == home_team)]
                match = match[(match['away_team'] == away_team) | (match['away'] == away_team)]
                if len(match) > 0:
                    match = match.iloc[-1]
                    feat = [[
                        match['form_diff_5'],
                        match['form_diff_10'],
                        match['form_diff_15'],
                        match['gd_diff_5'],
                        match['gd_diff_10'],
                        match['gd_diff_15'],
                        match.get('home_advantage', 0.46),
                        match.get('home_away_form_diff', 0),
                        match.get('league_pos_diff', 0)
                    ]]
            
            if feat is None:
                import duckdb
                conn = duckdb.connect("database/results.db")
                
                tables = [r[0] for r in conn.execute("SHOW TABLES").fetchall()]
                features_table = f"features_{predict_league}"
                
                if features_table in tables:
                    feat_df = conn.execute(f"""
                        SELECT * FROM {features_table} 
                        WHERE home_team = '{home_team}' AND away_team = '{away_team}'
                        ORDER BY match_date DESC LIMIT 1
                    """).df()
                    
                    if len(feat_df) > 0:
                        feat_row = feat_df.iloc[0]
                        feat = [[
                            feat_row['form_diff_5'],
                            feat_row['form_diff_10'],
                            feat_row['form_diff_15'],
                            feat_row['gd_diff_5'],
                            feat_row['gd_diff_10'],
                            feat_row['gd_diff_15'],
                            feat_row['home_advantage'],
                            feat_row['home_away_form_diff'],
                            feat_row['league_pos_diff']
                        ]]
                
                conn.close()
            
            prior = session_df if session_df is not None else df
            
            if predict_model == "Poisson":
                prior = None
                if session_df is not None and len(session_df) > 0 and 'home_goals' in session_df.columns:
                    prior = session_df
                else:
                    import duckdb
                    conn = duckdb.connect("database/results.db")
                    tables = [r[0] for r in conn.execute("SHOW TABLES").fetchall()]
                    feat_table = f"features_{predict_league}"
                    if feat_table in tables:
                        prior = conn.execute(f"SELECT * FROM {feat_table} ORDER BY match_date").df()
                    conn.close()
                if prior is None or len(prior) == 0:
                    prior = load_builtin(predict_league)
            
            if prior is not None and 'home' in prior.columns and 'home_team' not in prior.columns:
                prior = prior.rename(columns={'home': 'home_team', 'away': 'away_team', 'home_goals': 'home_goals', 'away_goals': 'away_goals'})
            
            if 'match_date' not in prior.columns and 'date' in prior.columns:
                prior = prior.rename(columns={'date': 'match_date'})
            
            if predict_model == "LR":
                if feat is not None:
                    lr_probs = model_info['model'].predict_proba(model_info['scaler'].transform(feat))[0]
                else:
                    st.warning("No features found for this match")
                    lr_probs = None
                poi_probs = None
            elif predict_model == "Poisson":
                lr_probs = None
                poi_probs = predict_poisson(home_team, away_team, prior)
            else:
                st.info("Bookmaker model requires historical match odds")
                lr_probs = None
                poi_probs = None
            
            if lr_probs is not None or poi_probs is not None:
                st.divider()
                st.subheader("Probabilities")
                
                if lr_probs is not None:
                    result_cols = st.columns(3)
                    with result_cols[0]:
                        st.metric("Home (LR)", f"{lr_probs[0]*100:.1f}%")
                    with result_cols[1]:
                        st.metric("Draw (LR)", f"{lr_probs[1]*100:.1f}%")
                    with result_cols[2]:
                        st.metric("Away (LR)", f"{lr_probs[2]*100:.1f}%")
                
                if poi_probs is not None:
                    poi_cols = st.columns(3)
                    with poi_cols[0]:
                        st.metric("Home (Poisson)", f"{poi_probs[0]*100:.1f}%")
                    with poi_cols[1]:
                        st.metric("Draw (Poisson)", f"{poi_probs[1]*100:.1f}%")
                    with poi_cols[2]:
                        st.metric("Away (Poisson)", f"{poi_probs[2]*100:.1f}%")
                
                probs_to_use = lr_probs if lr_probs is not None else poi_probs
                
                st.divider()
                st.subheader("Odds")
                
                margin = st.slider("Margin %", 0, 20, 5)
                
                fair_odds = [1/p if p > 0 else 0 for p in probs_to_use]
                margined_odds = [o / (1 + margin/100) for o in fair_odds]
                
                od1, od2, od3 = st.columns(3)
                with od1:
                    st.metric("Home", f"{margined_odds[0]:.2f}")
                with od2:
                    st.metric("Draw", f"{margined_odds[1]:.2f}")
                with od3:
                    st.metric("Away", f"{margined_odds[2]:.2f}")
                
                st.divider()
                st.subheader("Bookmaker Odds (optional)")
                
                bk_cols = st.columns(3)
                with bk_cols[0]:
                    bk_home = st.number_input("Home odds", min_value=1.0, value=2.0, step=0.01)
                with bk_cols[1]:
                    bk_draw = st.number_input("Draw odds", min_value=1.0, value=3.5, step=0.01)
                with bk_cols[2]:
                    bk_away = st.number_input("Away odds", min_value=1.0, value=4.0, step=0.01)
                
                if bk_home > 1:
                    implied_home = 1/bk_home
                    implied_draw = 1/bk_draw
                    implied_away = 1/bk_away
                    total = implied_home + implied_draw + implied_away
                    bookmaker_margin = (total - 1) * 100
                    
                    st.caption(f"Implied margin: {bookmaker_margin:.1f}%")
                    
                    ev_home = probs_to_use[0] * bk_home - 1
                    ev_draw = probs_to_use[1] * bk_draw - 1
                    ev_away = probs_to_use[2] * bk_away - 1
                    
                    st.subheader("Expected Value")
                    ev_cols = st.columns(3)
                    with ev_cols[0]:
                        st.metric("Home EV", f"{ev_home*100:.1f}%", delta_color="normal" if ev_home > 0 else "inverse")
                    with ev_cols[1]:
                        st.metric("Draw EV", f"{ev_draw*100:.1f}%", delta_color="normal" if ev_draw > 0 else "inverse")
                    with ev_cols[2]:
                        st.metric("Away EV", f"{ev_away*100:.1f}%", delta_color="normal" if ev_away > 0 else "inverse")