import requests
import pandas as pd
import numpy as np
import os
import joblib
from tensorflow import keras

YEAR = 2025  # Set to None for latest, or specify a year (e.g., 2025)

# --- Helper functions (reuse from pre_race_data.py logic) ---
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
    return data[0] if data else None

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

# --- Main prediction logic ---
def main():
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
    # --- Fetch latest race data ---
    session = get_latest_race_session()
    session_key = session['session_key']
    meeting_key = session['meeting_key']
    year = session['year']
    circuit = session.get('circuit_short_name', None)
    country = session.get('country_name', None)
    session_name = session.get('session_name', 'Race')
    print(f"Predicting for: {session_name} | {circuit} | {country} | {year}")
    grid = get_starting_grid(session_key)
    drivers = get_driver_info(session_key)
    driver_map = {d['driver_number']: d for d in drivers}
    meeting = get_meeting_info(meeting_key)
    weather = get_weather(meeting_key, session_key)
    # --- Compute driver/team form ---
    hist = get_historical_results_with_team()
    # Only use results before this race
    hist = hist[hist['session_key'] != session_key]
    # Build feature rows
    rows = []
    for entry in grid:
        drv_num = entry['driver_number']
        drv = driver_map.get(drv_num, {})
        row = {
            'grid_position': entry['position'],
            'qualifying_lap_time': entry.get('lap_duration', None),
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
        # Driver form
        drv_hist = hist[hist['driver_number'] == drv_num].sort_values('session_key')
        row['driver_form_last3'] = drv_hist['position'].shift(1).rolling(3, min_periods=1).mean().iloc[-1] if not drv_hist.empty else None
        # Team form
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
    # Predict top 5 probabilities
    top5_probs = model.predict(df[features]).flatten()
    df['top5_probability'] = top5_probs
    # Output top 5
    out = df.copy()
    out['driver'] = grid and [driver_map.get(g['driver_number'], {}).get('full_name', g['driver_number']) for g in grid] or None
    out['team'] = grid and [driver_map.get(g['driver_number'], {}).get('team_name', None) for g in grid] or None
    out = out[['driver', 'team', 'grid_position', 'top5_probability']]
    out = out.sort_values('top5_probability', ascending=False).reset_index(drop=True)
    print("\nTop 5 predicted finishers:")
    for i, row in out.head(5).iterrows():
        medal = ['🥇','🥈','🥉','🏅','🏅'][i] if i < 5 else ''
        print(f"{medal} {row['driver']} ({row['team']}) | Grid: {row['grid_position']} | Top 5 probability: {row['top5_probability']*100:.2f}%")
    print("\nFull top 5 probability table:")
    print(out.to_string(index=False))

def fetch_and_print_latest_official_results():
    import requests
    API = 'https://api.openf1.org/v1'
    # Get latest race session
    resp = requests.get(f'{API}/sessions?session_type=Race')
    try:
        sessions = resp.json()
    except Exception:
        print("Could not parse sessions response as JSON:", resp.text)
        return
    if not isinstance(sessions, list):
        print("Unexpected sessions response:", sessions)
        return
    if not sessions:
        print("No race sessions found.")
        return
    latest = sorted(sessions, key=lambda x: x['date_start'])[-1]
    session_key = latest['session_key']
    year = latest.get('year', '')
    circuit = latest.get('circuit_short_name', '')
    session_name = latest.get('session_name', 'Race')
    print(f"\nOfficial Results for: {session_name} | {circuit} | {year}")
    # Fetch results
    results = requests.get(f'{API}/session_result?session_key={session_key}').json()
    if not results:
        print("No results found for this session.")
        return

    # Try to get driver info for mapping
    driver_info = {}
    try:
        drivers = requests.get(f'{API}/drivers').json()
        for d in drivers:
            driver_info[str(d.get('driver_number'))] = d.get('full_name', '')
    except Exception:
        pass

    print(f"{'Pos':<4} {'Driver':<20} {'Team':<20} {'Time/Retired':<15}")
    for r in sorted(results, key=lambda x: int(x['position']) if str(x['position']).isdigit() else 99):
        pos = r.get('position', '') or ''
        driver = r.get('full_name') or driver_info.get(str(r.get('driver_number', '')), str(r.get('driver_number', ''))) or ''
        team = r.get('team_name', '') or ''
        time_ret = r.get('time') or r.get('retired') or ''
        print(f"{pos:<4} {driver:<20} {team:<20} {time_ret:<15}")

if __name__ == "__main__":
    main()
    fetch_and_print_latest_official_results() 