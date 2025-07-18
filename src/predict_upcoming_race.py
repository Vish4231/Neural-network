import requests
import pandas as pd
import numpy as np
import os
import joblib
from tensorflow import keras
from datetime import datetime, timezone
import argparse
import time

# Remove hardcoded YEAR = 2025
# Add argument parsing at the top of main()
def main():
    parser = argparse.ArgumentParser(description="Predict F1 race results.")
    parser.add_argument('--year', type=int, default=None, help='Year of the race (e.g., 2024)')
    parser.add_argument('--circuit', type=str, default=None, help='Circuit short name (e.g., Spa-Francorchamps)')
    parser.add_argument('--next', action='store_true', help='Predict the next scheduled race (future race)')
    parser.add_argument('--cutoff_circuit', type=str, default=None, help='Circuit short name to use as data cutoff (e.g., Silverstone)')
    # Add other arguments as needed
    args = parser.parse_args()
    YEAR = args.year
    CIRCUIT = args.circuit
    PREDICT_NEXT = args.next
    CUTOFF_CIRCUIT = args.cutoff_circuit
    # Use YEAR and CIRCUIT in all relevant logic below

    # --- Helper functions (reuse from pre_race_data.py logic) ---
    def robust_get(url, max_retries=3, sleep_sec=2):
        for attempt in range(max_retries):
            resp = requests.get(url)
            try:
                data = resp.json()
            except Exception:
                data = None
            if isinstance(data, dict) and 'error' in data and 'rate limit' in data.get('detail', '').lower():
                print(f"[INFO] Rate limit hit. Sleeping {sleep_sec} seconds and retrying ({attempt+1}/{max_retries})...")
                time.sleep(sleep_sec)
                continue
            return data
        print(f"[ERROR] Failed to get data from {url} after {max_retries} retries.")
        return None

    def get_latest_race_session():
        url = "https://api.openf1.org/v1/sessions?session_type=Race"
        if YEAR is not None:
            url += f"&year={YEAR}"
        resp = requests.get(url)
        sessions = resp.json()
        if not sessions:
            raise Exception(f"No race sessions found for year {YEAR}.")
        latest = sorted(sessions, key=lambda x: x['date_start'], reverse=True)[0]
        return latest

    def get_starting_grid(session_key):
        url = f"https://api.openf1.org/v1/starting_grid?session_key={session_key}"
        resp = requests.get(url)
        return resp.json() if resp.status_code == 200 else []

    def get_driver_info(session_key):
        url = f"https://api.openf1.org/v1/drivers?session_key={session_key}"
        resp = requests.get(url)
        return resp.json() if resp.status_code == 200 else []

    def get_meeting_info(meeting_key):
        url = f"https://api.openf1.org/v1/meetings?meeting_key={meeting_key}"
        resp = requests.get(url)
        data = resp.json()
        if isinstance(data, list):
            return data[0] if data else None
        else:
            print(f"Unexpected meeting info response: {data}")
            return None

    def get_weather(meeting_key, session_key):
        url = f"https://api.openf1.org/v1/weather?meeting_key={meeting_key}&session_key={session_key}"
        resp = requests.get(url)
        data = resp.json()
        if not data:
            return None
        df = pd.DataFrame(data)
        return {
            'air_temperature': df['air_temperature'].mean() if 'air_temperature' in df else None,
            'humidity': df['humidity'].mean() if 'humidity' in df else None,
            'rainfall': df['rainfall'].mean() if 'rainfall' in df else None,
            'track_temperature': df['track_temperature'].mean() if 'track_temperature' in df else None,
            'wind_speed': df['wind_speed'].mean() if 'wind_speed' in df else None,
        }

    def get_historical_results():
        # Use all past session_results for driver/team form
        url = "https://api.openf1.org/v1/session_result"
        resp = requests.get(url)
        if resp.status_code != 200:
            return pd.DataFrame()
        df = pd.DataFrame(resp.json())
        # Only keep valid integer positions
        df = df[df['position'].apply(lambda x: str(x).isdigit())]
        df['position'] = df['position'].astype(int)
        return df

    def get_historical_results_with_team():
        # Use all past session_results for driver/team form
        url = "https://api.openf1.org/v1/session_result"
        resp = requests.get(url)
        if resp.status_code != 200:
            return pd.DataFrame()
        df = pd.DataFrame(resp.json())
        # Only keep valid integer positions
        df = df[df['position'].apply(lambda x: str(x).isdigit())]
        df['position'] = df['position'].astype(int)
        # Add team_name by fetching driver info for each session_key
        session_keys = df['session_key'].unique()
        team_map = {}
        for sk in session_keys:
            try:
                drivers = get_driver_info(sk)
                for d in drivers:
                    team_map[(sk, d['driver_number'])] = d.get('team_name', None)
            except Exception as e:
                continue
        df['team_name'] = df.apply(lambda row: team_map.get((row['session_key'], row['driver_number']), None), axis=1)
        missing = df['team_name'].isna().sum()
        if missing > 0:
            print(f"Warning: {missing} historical results missing team_name.")
        return df

    def get_next_scheduled_race():
        url = "https://api.openf1.org/v1/sessions?session_type=Race"
        resp = requests.get(url)
        sessions = resp.json()
        now = datetime.now(timezone.utc)
        # Find the next session with a start date in the future
        future_sessions = [s for s in sessions if 'date_start' in s and datetime.fromisoformat(s['date_start'].replace('Z', '+00:00')) > now]
        if not future_sessions:
            raise Exception("No future scheduled races found.")
        next_session = sorted(future_sessions, key=lambda x: x['date_start'])[0]
        return next_session

    # --- Main prediction logic ---
    features = [
        'grid_position', 'qualifying_lap_time', 'air_temperature', 'humidity', 'rainfall',
        'track_temperature', 'wind_speed', 'team_name', 'driver_name', 'circuit', 'country_code',
        'driver_form_last3', 'team_form_last3', 'qualifying_gap_to_pole', 'teammate_grid_delta',
        'track_type', 'overtaking_difficulty',
        'driver_championship_position', 'team_championship_position', 'driver_points_season', 'team_points_season'
    ]
    # Load top 5 model
    import joblib
    model = keras.models.load_model('model/pre_race_model_top5.keras')
    encoders = joblib.load('model/encoders_top5.pkl')
    scaler = joblib.load('model/scaler_top5.pkl')
    # Load XGBoost model
    import xgboost as xgb
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model('model/xgb_top5.model')
    # --- Fetch sessions for the given year and filter by circuit ---
    API = 'https://api.openf1.org/v1'
    if PREDICT_NEXT:
        session = get_next_scheduled_race()
        YEAR = session['year']
        CIRCUIT = session.get('circuit_short_name', None)
    else:
    sessions_url = f"{API}/sessions?session_type=Race"
    if YEAR:
        sessions_url += f"&year={YEAR}"
    sessions = requests.get(sessions_url).json()
    if not isinstance(sessions, list):
        print(f"Unexpected sessions response: {sessions}")
        return
    session = None
    if CIRCUIT:
            # Try to match circuit_short_name (case-insensitive) for the requested year
        for s in sessions:
                if str(s.get('circuit_short_name', '')).lower() == CIRCUIT.lower() and (YEAR is None or s.get('year', None) == YEAR):
                session = s
                break
        if not session:
                # If not found, search all years for the requested circuit
                all_sessions_url = f"{API}/sessions?session_type=Race"
                all_sessions = requests.get(all_sessions_url).json()
                matching_sessions = [s for s in all_sessions if str(s.get('circuit_short_name', '')).lower() == CIRCUIT.lower()]
                if matching_sessions:
                    # Use the most recent session for lineup, but aggregate features from all years
                    session = sorted(matching_sessions, key=lambda x: x['year'], reverse=True)[0]
                    print(f"\n[INFO] Circuit '{CIRCUIT}' not found for year {YEAR}. Using latest available year {session['year']} for driver lineup, but aggregating features from all years for simulation of {YEAR}.")
                    # Overwrite year in session to requested year for simulation
                    session['year'] = YEAR
                else:
                    available = sorted(set(s.get('circuit_short_name','') for s in all_sessions))
            print(f"Circuit '{CIRCUIT}' not found for year {YEAR}. Available circuits:")
            for c in available:
                print(f"  - {c}")
            return
    else:
        # Default to latest session if no circuit specified
        session = sorted(sessions, key=lambda x: x['date_start'], reverse=True)[0]
    # Use 'session' for all further logic
    session_key = session['session_key']
    meeting_key = session['meeting_key']
    year = session['year']
    circuit = session.get('circuit_short_name', None)
    country = session.get('country_name', None)
    session_name = session.get('session_name', 'Race')
    print(f"Predicting for: {session_name} | {circuit} | {country} | {year}")
    # Find the cutoff session (by circuit name) in the specified year
    if CUTOFF_CIRCUIT:
        cutoff_session = None
        for s in sessions: # Use 'sessions' here, not 'year_sessions'
            if str(s.get('circuit_short_name', '')).lower() == CUTOFF_CIRCUIT.lower():
                cutoff_session = s
                break
        if not cutoff_session:
            print(f"Cutoff circuit '{CUTOFF_CIRCUIT}' not found for year {YEAR}. Using target race as cutoff.")
            cutoff_session = session
    else:
        cutoff_session = session
    cutoff_date = cutoff_session['date_start']
    # --- Improved missing data handling using all years for the circuit ---
    # Get all sessions for this circuit (all years)
    all_sessions_url = f"{API}/sessions?session_type=Race"
    all_sessions = robust_get(all_sessions_url)
    circuit_sessions = [s for s in all_sessions if str(s.get('circuit_short_name', '')).lower() == circuit.lower()]
    circuit_session_keys = [s['session_key'] for s in circuit_sessions]
    # Get all historical results for this circuit (all years)
    hist = get_historical_results_with_team()
    # Ensure required columns exist in hist and hist_circuit
    if 'session_key' not in hist.columns:
        print('[ERROR] session_key column missing from historical results. Cannot proceed with race-specific feature engineering.')
        return
    if 'team_name' not in hist.columns:
        print('[WARNING] team_name column missing from historical results. Filling with None.')
        hist['team_name'] = None
    # After filtering, ensure hist_circuit has required columns
    hist_circuit = hist[hist['session_key'].isin(circuit_session_keys)]
    if 'team_name' not in hist_circuit.columns:
        print('[WARNING] team_name column missing from circuit-specific historical results. Filling with None.')
        hist_circuit['team_name'] = None
    # Use the most recent session for driver lineup, but ensure it's for the correct year if possible
    most_recent_session = None
    for s in sorted(circuit_sessions, key=lambda x: (x['year'], x['date_start']), reverse=True):
        if s['year'] == YEAR:
            most_recent_session = s
            break
    if not most_recent_session:
        most_recent_session = sorted(circuit_sessions, key=lambda x: (x['year'], x['date_start']), reverse=True)[0] if circuit_sessions else session
    # Get drivers for the most recent session for this circuit and year
    drivers = get_driver_info(most_recent_session['session_key'])
    if not drivers:
        # Fallback: use the most recent available year for this circuit
        print(f"[WARNING] No driver lineup found for {circuit} {YEAR}. Using most recent available year for this circuit.")
        if circuit_sessions:
            fallback_session = sorted(circuit_sessions, key=lambda x: (x['year'], x['date_start']), reverse=True)[0]
            drivers = get_driver_info(fallback_session['session_key'])
    # --- Manual fallback for Imola 2025: use user-provided 2025 grid ---
    if (not drivers or len(drivers) == 0) and circuit.lower() in ["imola"] and YEAR == 2025:
        print("[MANUAL FALLBACK] No driver lineup found for Imola 2025. Using user-provided 2025 grid.")
        manual_lineup = [
            {"country": "United Kingdom", "driver_name": "Lando Norris", "team_name": "McLaren Mercedes", "driver_number": 4},
            {"country": "Australia", "driver_name": "Oscar Piastri", "team_name": "McLaren Mercedes", "driver_number": 81},
            {"country": "Germany", "driver_name": "Nico Hulkenberg", "team_name": "Kick Sauber Ferrari", "driver_number": 27},
            {"country": "United Kingdom", "driver_name": "Lewis Hamilton", "team_name": "Ferrari", "driver_number": 44},
            {"country": "Netherlands", "driver_name": "Max Verstappen", "team_name": "Red Bull Racing Honda RBPT", "driver_number": 1},
            {"country": "France", "driver_name": "Pierre Gasly", "team_name": "Alpine Renault", "driver_number": 10},
            {"country": "Canada", "driver_name": "Lance Stroll", "team_name": "Aston Martin Aramco Mercedes", "driver_number": 18},
            {"country": "Thailand", "driver_name": "Alexander Albon", "team_name": "Williams Mercedes", "driver_number": 23},
            {"country": "Spain", "driver_name": "Fernando Alonso", "team_name": "Aston Martin Aramco Mercedes", "driver_number": 14},
            {"country": "United Kingdom", "driver_name": "George Russell", "team_name": "Mercedes", "driver_number": 63},
            {"country": "United Kingdom", "driver_name": "Oliver Bearman", "team_name": "Haas Ferrari", "driver_number": 87},
            {"country": "Spain", "driver_name": "Carlos Sainz", "team_name": "Williams Mercedes", "driver_number": 55},
            {"country": "France", "driver_name": "Esteban Ocon", "team_name": "Haas Ferrari", "driver_number": 31},
            {"country": "Monaco", "driver_name": "Charles Leclerc", "team_name": "Ferrari", "driver_number": 16},
            {"country": "Japan", "driver_name": "Yuki Tsunoda", "team_name": "Racing Bulls Honda RBPT", "driver_number": 22},
            {"country": "Italy", "driver_name": "Kimi Antonelli", "team_name": "Mercedes", "driver_number": 12},
            {"country": "France", "driver_name": "Isack Hadjar", "team_name": "Racing Bulls Honda RBPT", "driver_number": 6},
            {"country": "Brazil", "driver_name": "Gabriel Bortoleto", "team_name": "Kick Sauber Ferrari", "driver_number": 5},
            {"country": "New Zealand", "driver_name": "Liam Lawson", "team_name": "Racing Bulls Honda RBPT", "driver_number": 30},
            {"country": "Argentina", "driver_name": "Franco Colapinto", "team_name": "Alpine Renault", "driver_number": 43},
        ]
        drivers = []
        for entry in manual_lineup:
            drivers.append({
                'driver_number': entry['driver_number'],
                'full_name': entry['driver_name'],
                'team_name': entry['team_name'],
                'country_code': entry['country'],
            })
    if not drivers:
        print(f"[ERROR] No driver lineup found for {circuit} in any year. Cannot make prediction.")
        return
    driver_map = {d['driver_number']: d for d in drivers}
    # Build grid using driver/circuit-specific historical average grid positions
    grid = []
    for d in drivers:
        drv_num = d['driver_number']
        # Use average grid position for this driver at this circuit
        if 'grid_position' in hist_circuit.columns:
            avg_grid = hist_circuit[hist_circuit['driver_number'] == drv_num]['grid_position'].mean()
        else:
            avg_grid = None
        if pd.isna(avg_grid):
            # Fallback: use average grid position for this driver across all circuits
            if 'grid_position' in hist.columns:
                avg_grid = hist[hist['driver_number'] == drv_num]['grid_position'].mean()
            else:
                avg_grid = None
        grid.append({'driver_number': drv_num, 'position': int(avg_grid) if not pd.isna(avg_grid) and avg_grid is not None else 10})
    # Use average weather for this circuit (all years)
    url = f"https://api.openf1.org/v1/weather?circuit_short_name={circuit}"
    resp = requests.get(url)
    data = resp.json() if resp.status_code == 200 else []
    if data:
        dfw = pd.DataFrame(data)
        # Only use weather data up to cutoff date
        if 'date' in dfw.columns:
            dfw = dfw[dfw['date'] <= cutoff_date]
        weather = {
            'air_temperature': dfw['air_temperature'].mean() if 'air_temperature' in dfw else None,
            'humidity': dfw['humidity'].mean() if 'humidity' in dfw else None,
            'rainfall': dfw['rainfall'].mean() if 'rainfall' in dfw else None,
            'track_temperature': dfw['track_temperature'].mean() if 'track_temperature' in dfw else None,
            'wind_speed': dfw['wind_speed'].mean() if 'wind_speed' in dfw else None,
        }
    else:
        weather = {'air_temperature': 22, 'humidity': 60, 'rainfall': 0, 'track_temperature': 30, 'wind_speed': 5}
    # For each driver, use circuit- and driver-specific historical averages for qualifying lap time and form
    rows = []
    for entry in grid:
        drv_num = entry['driver_number']
        drv = driver_map.get(drv_num, {})
        # Use average qualifying lap time for this driver at this circuit
        if 'qualifying_lap_time' in hist_circuit.columns:
            avg_qual_time = hist_circuit[hist_circuit['driver_number'] == drv_num]['qualifying_lap_time'].mean()
        else:
            avg_qual_time = None
        if pd.isna(avg_qual_time):
            if 'qualifying_lap_time' in hist.columns:
                avg_qual_time = hist[hist['driver_number'] == drv_num]['qualifying_lap_time'].mean()
            else:
                avg_qual_time = None
        row = {
            'grid_position': entry['position'],
            'qualifying_lap_time': avg_qual_time,
            'air_temperature': weather['air_temperature'] if weather else None,
            'humidity': weather['humidity'] if weather else None,
            'rainfall': weather['rainfall'] if weather else None,
            'track_temperature': weather['track_temperature'] if weather else None,
            'wind_speed': weather['wind_speed'] if weather else None,
            'team_name': drv.get('team_name', None),
            'driver_name': drv.get('full_name', None),
            'circuit': circuit,
            'country_code': drv.get('country_code', None),
        }
        # Driver form: use all historical results for this driver at this circuit
        if 'position' in hist_circuit.columns:
            drv_hist = hist_circuit[hist_circuit['driver_number'] == drv_num].sort_values('session_key')
        else:
            drv_hist = pd.DataFrame()
        if drv_hist.empty and 'position' in hist.columns:
        drv_hist = hist[hist['driver_number'] == drv_num].sort_values('session_key')
        row['driver_form_last3'] = drv_hist['position'].shift(1).rolling(3, min_periods=1).mean().iloc[-1] if not drv_hist.empty else None
        # Team form: use all historical results for this team at this circuit
        team = drv.get('team_name', None)
        if 'position' in hist_circuit.columns:
            team_hist = hist_circuit[hist_circuit['team_name'] == team].sort_values('session_key')
        else:
            team_hist = pd.DataFrame()
        if team_hist.empty and 'position' in hist.columns:
        team_hist = hist[hist['team_name'] == team].sort_values('session_key')
        row['team_form_last3'] = team_hist['position'].shift(1).rolling(3, min_periods=1).mean().iloc[-1] if not team_hist.empty else None
        # Advanced features (fill with None or compute if possible)
        row['qualifying_gap_to_pole'] = None
        row['teammate_grid_delta'] = None
        row['track_type'] = 'permanent'  # default
        row['overtaking_difficulty'] = 3  # default
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        print("[ERROR] No driver lineup or data available for this race. Cannot make prediction.")
        return
    # Ensure all required features are present
    for col in ['driver_championship_position', 'team_championship_position', 'driver_points_season', 'team_points_season']:
        if col not in df.columns:
            df[col] = -1
    # Ensure all required feature columns exist in df
    features = [
        'grid_position', 'qualifying_lap_time', 'air_temperature', 'humidity', 'rainfall',
        'track_temperature', 'wind_speed', 'team_name', 'driver_name', 'circuit', 'country_code',
        'driver_form_last3', 'team_form_last3', 'qualifying_gap_to_pole', 'teammate_grid_delta',
        'track_type', 'overtaking_difficulty',
        'driver_championship_position', 'team_championship_position', 'driver_points_season', 'team_points_season'
    ]
    cat_features = ['team_name', 'driver_name', 'circuit', 'country_code', 'track_type']
    for col in features:
        if col not in df.columns:
            if col in cat_features:
                print(f"[WARNING] {col} column missing from prediction DataFrame. Filling with 'Unknown'.")
                df[col] = 'Unknown'
            else:
                print(f"[WARNING] {col} column missing from prediction DataFrame. Filling with -1.")
                df[col] = -1
    # Now safe to encode categoricals and scale numerics
    for col in cat_features:
        le = encoders[col]
        # Map unseen labels to the first class (acts as 'unknown')
        df[col] = df[col].astype(str).apply(lambda x: x if x in le.classes_ else le.classes_[0])
        df[col] = le.transform(df[col])
    num_features = [f for f in features if f not in cat_features]
    df[num_features] = scaler.transform(df[num_features])
    # Fill missing championship/points features with -1
    for col in ['driver_championship_position', 'team_championship_position', 'driver_points_season', 'team_points_season']:
        if col in df.columns:
            df[col] = df[col].fillna(-1)
    # Fill all NaNs in features with -1
    df[features] = df[features].fillna(-1)
    # Ensure all numeric columns are float
    for col in features:
        if col not in cat_features:
            df[col] = df[col].astype(float)
    # Debug: print NaN status and dtypes
    print("Any NaNs in features before prediction?", df[features].isnull().any().any())
    print("Columns with NaNs:", df[features].columns[df[features].isnull().any()].tolist())
    print(df[features].dtypes)
    print(df[features].head())
    # Predict top 5 probabilities
    # Predict with neural net
    nn_probs = model.predict(df[features]).flatten()
    # Predict with XGBoost
    xgb_probs = xgb_model.predict_proba(df[features])[:,1]
    # Predict with LightGBM
    import lightgbm as lgb
    lgbm_model = lgb.Booster(model_file='model/lgbm_top5.txt')
    lgbm_probs = lgbm_model.predict(df[features])
    # Predict with CatBoost
    import catboost as cb
    cat_model = cb.CatBoostClassifier()
    cat_model.load_model('model/catboost_top5.cbm')
    cat_probs = cat_model.predict_proba(df[features])[:,1]
    # Ensemble: average probabilities
    ensemble_probs = (nn_probs + xgb_probs + lgbm_probs + cat_probs) / 4
    # Max-prob (best possibility) ensemble
    max_probs = np.max(np.vstack([nn_probs, xgb_probs, lgbm_probs, cat_probs]), axis=0)
    df['top5_probability_nn'] = nn_probs
    df['top5_probability_xgb'] = xgb_probs
    df['top5_probability_lgbm'] = lgbm_probs
    df['top5_probability_cat'] = cat_probs
    df['top5_probability_ensemble'] = ensemble_probs
    df['top5_probability_max'] = max_probs
    # Output top 5 for each
    out = df.copy()
    out['driver'] = grid and [driver_map.get(g['driver_number'], {}).get('full_name', g['driver_number']) for g in grid] or None
    out['team'] = grid and [driver_map.get(g['driver_number'], {}).get('team_name', None) for g in grid] or None
    out = out[['driver', 'team', 'grid_position', 'top5_probability_nn', 'top5_probability_xgb', 'top5_probability_lgbm', 'top5_probability_cat', 'top5_probability_ensemble', 'top5_probability_max']]
    # Print combined best possibility (max-prob) top 5
    out_max = out.sort_values('top5_probability_max', ascending=False).reset_index(drop=True)
    print("\nTop 5 predicted finishers (Combined Best Possibility - Max Probability):")
    for idx, row in enumerate(out_max.itertuples(), 1):
        medal = ["🥇", "🥈", "🥉", "🏅", "🏅"][idx-1] if idx <= 5 else ""
        print(f"{idx}. {medal} {row.driver} ({row.team}) | Grid: {row.grid_position} | Max Top 5 probability: {row.top5_probability_max:.2%}")
        if idx == 5:
            break
    print("\nFull top 5 probability table (Combined Best Possibility):")
    print(out_max.to_string(index=False))
    # Print ensemble average top 5
    out_ens = out.sort_values('top5_probability_ensemble', ascending=False).reset_index(drop=True)
    print("\nTop 5 predicted finishers (Ensemble Average):")
    for idx, row in enumerate(out_ens.itertuples(), 1):
        medal = ["🥇", "🥈", "🥉", "🏅", "🏅"][idx-1] if idx <= 5 else ""
        print(f"{idx}. {medal} {row.driver} ({row.team}) | Grid: {row.grid_position} | Ensemble Top 5 probability: {row.top5_probability_ensemble:.2%}")
        if idx == 5:
            break
    # Optionally, print top 5s for each model
    print("\nTop 5 predicted finishers (Neural Net only):")
    print(out.sort_values('top5_probability_nn', ascending=False).head(5)[['driver','team','grid_position','top5_probability_nn']])
    print("\nTop 5 predicted finishers (XGBoost only):")
    print(out.sort_values('top5_probability_xgb', ascending=False).head(5)[['driver','team','grid_position','top5_probability_xgb']])
    print("\nTop 5 predicted finishers (LightGBM only):")
    print(out.sort_values('top5_probability_lgbm', ascending=False).head(5)[['driver','team','grid_position','top5_probability_lgbm']])
    print("\nTop 5 predicted finishers (CatBoost only):")
    print(out.sort_values('top5_probability_cat', ascending=False).head(5)[['driver','team','grid_position','top5_probability_cat']])

    # After predictions, only fetch and print actual results if the race has already happened
    race_date = None
    try:
        # Try to get the race date from the session or meeting info
        if 'session' in locals() and session is not None and 'date_start' in session:
            race_date = session['date_start']
        elif 'latest' in locals() and latest is not None and 'date_start' in latest:
            race_date = latest['date_start']
    except Exception:
        pass

    show_actual = False
    if race_date:
        try:
            race_dt = datetime.fromisoformat(race_date.replace('Z', '+00:00'))
            now = datetime.now(timezone.utc)
            if race_dt < now:
                show_actual = True
        except Exception:
            pass

    if show_actual:
        # Fetch and print actual results as before, but map driver_number to name/team
        try:
            API = 'https://api.openf1.org/v1'
            session_key = session['session_key'] if 'session' in locals() and session is not None else None
            if session_key:
                results = requests.get(f'{API}/session_result?session_key={session_key}').json()
                # Fetch driver info for mapping
                driver_info = {}
                team_info = {}
                try:
                    drivers = requests.get(f'{API}/drivers?session_key={session_key}').json()
                    if isinstance(drivers, list):
                        for d in drivers:
                            if isinstance(d, dict):
                                driver_info[str(d.get('driver_number'))] = d.get('full_name', '')
                                team_info[str(d.get('driver_number'))] = d.get('team_name', '')
                            else:
                                print(f"Unexpected driver entry: {d}")
                    else:
                        print(f"Unexpected drivers response: {drivers}")
                except Exception as e:
                    print(f"Error fetching drivers: {e}")
                if results:
                    print("\nActual top 5 finishers:")
                    # Sort by position (handle int and string like 'DQ')
                    def pos_key(x):
                        try:
                            return int(x['position'])
                        except Exception as e:
                            print(f"Non-integer position in results: {x.get('position')}, error: {e}")
                            return 99
                    for idx, row in enumerate(sorted(results, key=pos_key)[:5], 1):
                        if not isinstance(row, dict):
                            print(f"Unexpected result row: {row}")
                            continue
                        driver_num = str(row.get('driver_number', ''))
                        driver = driver_info.get(driver_num, f"#{driver_num}")
                        team = team_info.get(driver_num, 'Unknown')
                        grid = row.get('grid_position', 'Unknown')
                        pos = row.get('position', 'Unknown')
                        print(f"{idx}. {driver} ({team}) | Grid: {grid} | Position: {pos}")
                else:
                    print("\nActual results not available for this race.")
        except Exception as e:
            print(f"\nCould not fetch official results: {e}")
    else:
        print("\nThis race has not happened yet. Only showing prediction based on previous data.")

if __name__ == "__main__":
    main() 