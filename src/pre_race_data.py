import requests
import pandas as pd
import os
from tqdm import tqdm

# Helper to get all race sessions for all years
START_YEAR = 2018
END_YEAR = 2024  # update as needed

def get_race_sessions():
    sessions = []
    for year in range(START_YEAR, END_YEAR+1):
        url = f"https://api.openf1.org/v1/sessions?year={year}&session_type=Race"
        resp = requests.get(url)
        if resp.status_code != 200:
            continue
        for s in resp.json():
            sessions.append(s)
    return sessions

def get_starting_grid(session_key):
    url = f"https://api.openf1.org/v1/starting_grid?session_key={session_key}"
    resp = requests.get(url)
    if resp.status_code != 200:
        return []
    return resp.json()

def get_session_result(session_key):
    url = f"https://api.openf1.org/v1/session_result?session_key={session_key}"
    resp = requests.get(url)
    if resp.status_code != 200:
        return []
    return resp.json()

def get_driver_info(session_key):
    url = f"https://api.openf1.org/v1/drivers?session_key={session_key}"
    resp = requests.get(url)
    if resp.status_code != 200:
        return []
    return resp.json()

def get_meeting_info(meeting_key):
    url = f"https://api.openf1.org/v1/meetings?meeting_key={meeting_key}"
    resp = requests.get(url)
    if resp.status_code != 200:
        return None
    data = resp.json()
    return data[0] if data else None

def get_weather(meeting_key, session_key):
    url = f"https://api.openf1.org/v1/weather?meeting_key={meeting_key}&session_key={session_key}"
    resp = requests.get(url)
    if resp.status_code != 200:
        return None
    data = resp.json()
    if not data:
        return None
    # Use mean of available weather samples
    df = pd.DataFrame(data)
    return {
        'air_temperature': df['air_temperature'].mean() if 'air_temperature' in df else None,
        'humidity': df['humidity'].mean() if 'humidity' in df else None,
        'rainfall': df['rainfall'].mean() if 'rainfall' in df else None,
        'track_temperature': df['track_temperature'].mean() if 'track_temperature' in df else None,
        'wind_speed': df['wind_speed'].mean() if 'wind_speed' in df else None,
    }

def main():
    os.makedirs('data', exist_ok=True)
    sessions = get_race_sessions()
    all_rows = []
    for sess in tqdm(sessions, desc="Races"):
        session_key = sess['session_key']
        meeting_key = sess['meeting_key']
        year = sess['year']
        circuit = sess.get('circuit_short_name', None)
        country = sess.get('country_name', None)
        grid = get_starting_grid(session_key)
        if not grid:
            continue
        results = get_session_result(session_key)
        if not results:
            continue
        drivers = get_driver_info(session_key)
        driver_map = {d['driver_number']: d for d in drivers}
        meeting = get_meeting_info(meeting_key)
        weather = get_weather(meeting_key, session_key)
        # Map finishing position by driver_number
        pos_map = {r['driver_number']: r['position'] for r in results}
        for entry in grid:
            drv_num = entry['driver_number']
            row = {
                'year': year,
                'circuit': circuit,
                'country': country,
                'driver_number': drv_num,
                'grid_position': entry['position'],
                'qualifying_lap_time': entry.get('lap_duration', None),
                'finishing_position': pos_map.get(drv_num, None),
            }
            # Add driver/team info
            drv = driver_map.get(drv_num, {})
            row['team_name'] = drv.get('team_name', None)
            row['driver_name'] = drv.get('full_name', None)
            row['country_code'] = drv.get('country_code', None)
            # Add weather
            if weather:
                row.update(weather)
            all_rows.append(row)
    df = pd.DataFrame(all_rows)
    df = df.dropna(subset=['finishing_position', 'grid_position', 'team_name'])
    df.to_csv('data/pre_race_features.csv', index=False)
    print(f"Saved {len(df)} rows to data/pre_race_features.csv")

if __name__ == "__main__":
    main() 