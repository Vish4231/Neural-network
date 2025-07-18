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
    # Find all sessions for the specified year, sorted by date
    year_sessions_url = f"{API}/sessions?session_type=Race&year={YEAR}"
    year_sessions = requests.get(year_sessions_url).json()
    if not isinstance(year_sessions, list):
        print(f"Unexpected sessions response: {year_sessions}")
        return
    # Find the specified race in the year
    target_session = None
    for s in year_sessions:
        if str(s.get('circuit_short_name', '')).lower() == circuit.lower():
            target_session = s
            break
    if not target_session:
        # If not found, fallback to previous logic (simulate future race)
        all_sessions_url = f"{API}/sessions?session_type=Race"
        all_sessions = requests.get(all_sessions_url).json()
        # Ensure all_sessions is a list of dicts
        if not isinstance(all_sessions, list) or not all_sessions or not isinstance(all_sessions[0], dict):
            print(f"[ERROR] Unexpected response from sessions API: {all_sessions}")
            return
        matching_sessions = [s for s in all_sessions if str(s.get('circuit_short_name', '')).lower() == circuit.lower()]
        if matching_sessions:
            target_session = sorted(matching_sessions, key=lambda x: (x['year'], x['date_start']), reverse=True)[0]
            print(f"\n[INFO] Circuit '{circuit}' not found for year {YEAR}. Using latest available year {target_session['year']} for driver lineup, but aggregating features from all years up to {YEAR} for simulation of {YEAR}.")
            target_session['year'] = YEAR
        else:
            available = sorted(set(s.get('circuit_short_name','') for s in all_sessions))
            print(f"Circuit '{circuit}' not found for year {YEAR}. Available circuits:")
            for c in available:
                print(f"  - {c}")
            return
    # Find the cutoff session (by circuit name) in the specified year
    if CUTOFF_CIRCUIT:
        cutoff_session = None
        for s in year_sessions:
            if str(s.get('circuit_short_name', '')).lower() == CUTOFF_CIRCUIT.lower():
                cutoff_session = s
                break
        if not cutoff_session:
            print(f"Cutoff circuit '{CUTOFF_CIRCUIT}' not found for year {YEAR}. Using target race as cutoff.")
            cutoff_session = target_session
    else:
        cutoff_session = target_session
    cutoff_date = cutoff_session['date_start']
    # Gather all sessions up to and including the cutoff race (all years before, and current year up to cutoff)
    all_sessions_url = f"{API}/sessions?session_type=Race"
    all_sessions = robust_get(all_sessions_url)
    # Sessions before the specified year
    sessions_before = [s for s in all_sessions if s['year'] < YEAR]
    # Sessions in the specified year up to and including the cutoff race
    sessions_this_year = [s for s in all_sessions if s['year'] == YEAR and s['date_start'] <= cutoff_date]
    used_sessions = sessions_before + sessions_this_year
    used_session_keys = [s['session_key'] for s in used_sessions if 'session_key' in s]
    # Get all historical results up to and including the specified race
    hist = get_historical_results_with_team()
    hist = hist[hist['session_key'].isin(used_session_keys)]
    # Use the driver lineup from the cutoff session in the specified year if available
    driver_lineup_session = None
    # Try to find the session for the cutoff circuit in the specified year
    for s in year_sessions:
        if str(s.get('circuit_short_name', '')).lower() == (CUTOFF_CIRCUIT or circuit).lower():
            driver_lineup_session = s
            break
    if not driver_lineup_session:
        print(f"[WARNING] No session found for {CUTOFF_CIRCUIT or circuit} {YEAR}. Trying to use target session.")
        driver_lineup_session = target_session
    drivers = robust_get(f"https://api.openf1.org/v1/drivers?session_key={driver_lineup_session['session_key']}")
    if not drivers:
        # Fallback: use the most recent available year for this circuit
        all_sessions_url = f"{API}/sessions?session_type=Race"
        all_sessions = robust_get(all_sessions_url)
        if not isinstance(all_sessions, list) or not all_sessions or not isinstance(all_sessions[0], dict):
            print(f"[ERROR] Unexpected response from sessions API: {all_sessions}")
            return
        matching_sessions = [s for s in all_sessions if isinstance(s, dict) and str(s.get('circuit_short_name', '')).lower() == circuit.lower()]
        if matching_sessions:
            fallback_session = sorted(matching_sessions, key=lambda x: (x['year'], x['date_start']), reverse=True)[0]
            print(f"[WARNING] No driver lineup found for {circuit} {YEAR} (cutoff: {CUTOFF_CIRCUIT or circuit}). The OpenF1 API does not have {YEAR} entry data yet. Using latest available year {fallback_session['year']} for driver lineup as a fallback.")
            drivers = robust_get(f"https://api.openf1.org/v1/drivers?session_key={fallback_session['session_key']}")
        else:
            print(f"[ERROR] No driver lineup found for {circuit} in any year.")
            return
    driver_map = {d['driver_number']: d for d in drivers}
    # Build grid using average grid positions up to and including the specified race
    grid = []
    for d in drivers:
        drv_num = d['driver_number']
        avg_grid = hist[hist['driver_number'] == drv_num]['grid_position'].mean() if 'grid_position' in hist.columns else None
        grid.append({'driver_number': drv_num, 'position': int(avg_grid) if avg_grid is not None else 10})
    # Use average weather for this circuit up to and including the specified race
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
    # For each driver, use average qualifying lap time up to and including the specified race
    rows = []
    for entry in grid:
        drv_num = entry['driver_number']
        drv = driver_map.get(drv_num, {})
        avg_qual_time = None
        if 'qualifying_lap_time' in hist.columns:
            q_times = hist[hist['driver_number'] == drv_num]['qualifying_lap_time']
            q_times = q_times[pd.notnull(q_times)]
            if not q_times.empty:
                avg_qual_time = q_times.mean()
        row = {
            'grid_position': entry['position'],
            'qualifying_lap_time': entry.get('lap_duration', avg_qual_time),
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
        # Driver form: use all historical results for this driver up to and including the specified race
        drv_hist = hist[hist['driver_number'] == drv_num].sort_values('session_key')
        row['driver_form_last3'] = drv_hist['position'].shift(1).rolling(3, min_periods=1).mean().iloc[-1] if not drv_hist.empty else None
        # Team form: use all historical results for this team up to and including the specified race
        team = drv.get('team_name', None)
        team_hist = hist[hist['team_name'] == team].sort_values('session_key')
        row['team_form_last3'] = team_hist['position'].shift(1).rolling(3, min_periods=1).mean().iloc[-1] if not team_hist.empty else None
        # Advanced features (fill with None or compute if possible)
        row['qualifying_gap_to_pole'] = None
        row['teammate_grid_delta'] = None
        row['track_type'] = 'permanent'  # default
        row['overtaking_difficulty'] = 3  # default
        rows.append(row)
    df = pd.DataFrame(rows)
    # Ensure all required features are present
    for col in ['driver_championship_position', 'team_championship_position', 'driver_points_season', 'team_points_season']:
        if col not in df.columns:
            df[col] = -1
    # Now safe to encode categoricals and scale numerics
    cat_features = ['team_name', 'driver_name', 'circuit', 'country_code', 'track_type']
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
    # Ensemble: average probabilities
    ensemble_probs = (nn_probs + xgb_probs) / 2
    df['top5_probability_nn'] = nn_probs
    df['top5_probability_xgb'] = xgb_probs
    df['top5_probability_ensemble'] = ensemble_probs
    # Output top 5 for each
    out = df.copy()
    out['driver'] = grid and [driver_map.get(g['driver_number'], {}).get('full_name', g['driver_number']) for g in grid] or None
    out['team'] = grid and [driver_map.get(g['driver_number'], {}).get('team_name', None) for g in grid] or None
    out = out[['driver', 'team', 'grid_position', 'top5_probability_nn', 'top5_probability_xgb', 'top5_probability_ensemble']]
    out = out.sort_values('top5_probability_ensemble', ascending=False).reset_index(drop=True)
    print("\nTop 5 predicted finishers (Ensemble):")
    for idx, row in enumerate(out.itertuples(), 1):
        medal = ["🥇", "🥈", "🥉", "🏅", "🏅"][idx-1] if idx <= 5 else ""
        print(f"{idx}. {medal} {row.driver} ({row.team}) | Grid: {row.grid_position} | Ensemble Top 5 probability: {row.top5_probability_ensemble:.2%}")
    print("\nFull top 5 probability table (Ensemble):")
    print(out.to_string(index=False))
    # Optionally, print top 5s for each model
    print("\nTop 5 predicted finishers (Neural Net only):")
    print(out.sort_values('top5_probability_nn', ascending=False).head(5)[['driver','team','grid_position','top5_probability_nn']])
    print("\nTop 5 predicted finishers (XGBoost only):")
    print(out.sort_values('top5_probability_xgb', ascending=False).head(5)[['driver','team','grid_position','top5_probability_xgb']])

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