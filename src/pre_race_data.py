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
    # Add driver form: average finish last 3 races (prior to current)
    df = df.sort_values(['driver_number', 'year', 'circuit'])
    df['finishing_position_int'] = df['finishing_position'].astype(int)
    df['driver_form_last3'] = (
        df.groupby('driver_number')['finishing_position_int']
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    )
    # Add team form: average finish last 3 races (prior to current)
    df['team_form_last3'] = (
        df.groupby('team_name')['finishing_position_int']
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    )
    # --- Add advanced features ---
    # Qualifying gap to pole
    df['qualifying_gap_to_pole'] = None
    for (year, circuit), group in df.groupby(['year', 'circuit']):
        try:
            pole_time = group['qualifying_lap_time'].min()
            mask = (df['year'] == year) & (df['circuit'] == circuit)
            df.loc[mask, 'qualifying_gap_to_pole'] = df.loc[mask, 'qualifying_lap_time'] - pole_time
        except Exception:
            pass
    # Teammate grid delta
    df['teammate_grid_delta'] = None
    for (year, circuit, team), group in df.groupby(['year', 'circuit', 'team_name']):
        if len(group) < 2:
            continue
        for idx, row in group.iterrows():
            teammate_rows = group[group['driver_number'] != row['driver_number']]
            if not teammate_rows.empty:
                teammate_grid = teammate_rows['grid_position'].values[0]
                df.at[idx, 'teammate_grid_delta'] = row['grid_position'] - teammate_grid
    # Championship position and points (driver/team)
    df['driver_championship_position'] = None
    df['team_championship_position'] = None
    df['driver_points_season'] = None
    df['team_points_season'] = None
    for (year, circuit), group in df.groupby(['year', 'circuit']):
        # Fetch standings up to this race (not including this race)
        try:
            import requests
            # Get all sessions for this year
            sessions = requests.get(f'https://api.openf1.org/v1/sessions?year={year}&session_type=Race').json()
            sessions = sorted(sessions, key=lambda x: x['date_start'])
            this_session = [s for s in sessions if s.get('circuit_short_name') == circuit]
            if not this_session:
                continue
            idx = sessions.index(this_session[0])
            if idx == 0:
                continue  # No prior races
            prev_sessions = sessions[:idx]
            # Get last session_key before this race
            last_session_key = prev_sessions[-1]['session_key']
            # Fetch driver standings
            standings = requests.get(f'https://api.openf1.org/v1/driver_standings?session_key={last_session_key}').json()
            driver_standings = {s['driver_number']: s for s in standings}
            # Fetch constructor standings
            constructors = requests.get(f'https://api.openf1.org/v1/constructor_standings?session_key={last_session_key}').json()
            team_standings = {s['team_name']: s for s in constructors}
            mask = (df['year'] == year) & (df['circuit'] == circuit)
            for idx2, row in df[mask].iterrows():
                drv_num = row['driver_number']
                team = row['team_name']
                drv_stand = driver_standings.get(str(drv_num))
                if drv_stand:
                    df.at[idx2, 'driver_championship_position'] = drv_stand.get('position')
                    df.at[idx2, 'driver_points_season'] = drv_stand.get('points')
                team_stand = team_standings.get(team)
                if team_stand:
                    df.at[idx2, 'team_championship_position'] = team_stand.get('position')
                    df.at[idx2, 'team_points_season'] = team_stand.get('points')
        except Exception as e:
            print(f"Warning: Could not fetch standings for {year} {circuit}: {e}")
            continue
    # Track type & overtaking difficulty (static mapping)
    track_type_map = {'Monaco': 'street', 'Baku': 'street', 'Singapore': 'street', 'Jeddah': 'street'}
    overtaking_map = {'Monaco': 1, 'Baku': 4, 'Singapore': 2, 'Jeddah': 5}  # Example
    df['track_type'] = df['circuit'].map(track_type_map).fillna('permanent')
    df['overtaking_difficulty'] = df['circuit'].map(overtaking_map).fillna(3)
    # Weather forecast (if available)
    df['weather_rain_forecast'] = None
    # Filter out rows where finishing_position is not a valid integer (e.g., DQ, DNS, DNF)
    def is_int_str(x):
        try:
            int(x)
            return True
        except:
            return False
    invalid_rows = ~df['finishing_position'].apply(is_int_str)
    if invalid_rows.any():
        print(f"Dropping {invalid_rows.sum()} rows with non-integer finishing_position values: {df.loc[invalid_rows, 'finishing_position'].unique().tolist()}")
        df = df[~invalid_rows]
    df['finishing_position'] = df['finishing_position'].astype(int)
    df['finishing_position_int'] = df['finishing_position']
    df.to_csv('data/pre_race_features.csv', index=False)
    print(f"Saved {len(df)} rows to data/pre_race_features.csv")

if __name__ == "__main__":
    main() 