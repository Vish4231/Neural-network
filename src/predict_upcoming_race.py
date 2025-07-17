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
    # Load encoders and scaler from training (retrain to save them if needed)
    # For now, re-fit on all data as in training script (for demo)
    # In production, save/load encoders/scaler with joblib
    model = keras.models.load_model('model/pre_race_model.keras')
    # Load training data to re-fit encoders/scaler
    train_df = pd.read_csv('data/pre_race_features.csv')
    cat_features = ['team_name', 'driver_name', 'circuit', 'country_code']
    num_features = [
        'grid_position', 'qualifying_lap_time', 'air_temperature', 'humidity', 'rainfall',
        'track_temperature', 'wind_speed', 'driver_form_last3', 'team_form_last3'
    ]
    features = num_features + cat_features
    # Fit encoders
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    encoders = {}
    for col in cat_features:
        le = LabelEncoder()
        train_df[col] = train_df[col].astype(str)
        le.fit(train_df[col])
        encoders[col] = le
    scaler = StandardScaler()
    scaler.fit(train_df[num_features])

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
        rows.append(row)
    df = pd.DataFrame(rows)
    # Encode categorical
    for col in cat_features:
        df[col] = df[col].astype(str)
        df[col] = df[col].apply(lambda x: x if x in encoders[col].classes_ else encoders[col].classes_[0])
        df[col] = encoders[col].transform(df[col])
    # Scale numeric
    df[num_features] = scaler.transform(df[num_features])
    # Predict
    preds = np.argmax(model.predict(df[features]), axis=1) + 1
    df['predicted_finish'] = preds
    # Output
    out = df.copy()
    out['grid_position'] = grid and [g['position'] for g in grid] or None
    out['driver'] = grid and [driver_map.get(g['driver_number'], {}).get('full_name', g['driver_number']) for g in grid] or None
    out['team'] = grid and [driver_map.get(g['driver_number'], {}).get('team_name', None) for g in grid] or None
    out = out[['predicted_finish', 'driver', 'team', 'grid_position']]
    out = out.sort_values('predicted_finish').reset_index(drop=True)
    out['predicted_finish'] = out.index + 1  # Ensure 1-N order in output
    # Add podium emojis
    podium_emojis = ['🥇', '🥈', '🥉'] + [''] * (len(out) - 3)
    out['podium'] = podium_emojis
    # Reorder columns for clarity
    out = out[['predicted_finish', 'podium', 'driver', 'team', 'grid_position']]
    print("Predicted finishing order for the upcoming race:")
    print(out.to_string(index=False))

if __name__ == "__main__":
    main() 