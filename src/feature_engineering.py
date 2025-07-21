import pandas as pd
import numpy as np
import os

def time_to_seconds(t):
    if pd.isnull(t) or t == '' or t == 'DNS' or t == 'DNF':
        return np.nan
    if isinstance(t, (int, float)):
        return t
    import re
    m = re.match(r'^(\d+):(\d+\.\d+)$', str(t).strip())
    if m:
        mins, secs = m.groups()
        return float(mins) * 60 + float(secs)
    try:
        return float(t)
    except:
        return np.nan

def add_rolling_form(df, form_window=3, form_col='driver_form_last3', group_col='driverId'):
    df = df.sort_values(['year', 'raceId'])
    df[form_col] = (
        df.groupby(group_col)['positionOrder']
        .apply(lambda x: x.shift(1).rolling(form_window, min_periods=1).mean())
        .reset_index(level=0, drop=True)
    )
    return df

def load_and_engineer_features():
    """
    Loads all historical data from the 'archive (1)' directory,
    engineers a comprehensive set of features, and returns a
    single DataFrame ready for model training and prediction.
    """
    archive_path = 'archive (1)/'
    results = pd.read_csv(archive_path + 'results.csv')
    races = pd.read_csv(archive_path + 'races.csv')
    drivers = pd.read_csv(archive_path + 'drivers.csv')
    constructors = pd.read_csv(archive_path + 'constructors.csv')
    qualifying = pd.read_csv(archive_path + 'qualifying.csv')
    pit_stops = pd.read_csv(archive_path + 'pit_stops.csv')
    lap_times = pd.read_csv(archive_path + 'lap_times.csv')

    # Merge results with races to get year, circuit, date
    results = results.merge(races[['raceId','year','name','circuitId','date']], on='raceId', how='left')
    results = results.merge(drivers[['driverId','driverRef','surname','forename','nationality']], on='driverId', how='left')
    results = results.merge(constructors[['constructorId','name']], on='constructorId', how='left', suffixes=('', '_team'))
    results = results.rename(columns={'name':'circuit', 'name_team':'team_name', 'surname':'driver_surname', 'forename':'driver_forename'})

    # Merge qualifying times (Q1/Q2/Q3)
    qual_cols = ['raceId','driverId','constructorId','q1','q2','q3','position']
    qualifying = qualifying[qual_cols]
    results = results.merge(qualifying, on=['raceId','driverId','constructorId'], how='left', suffixes=('', '_qual'))

    # Convert quali times to seconds
    for col in ['q1','q2','q3']:
        results[col] = results[col].apply(time_to_seconds)

    # Pit stop count
    pit_counts = pit_stops.groupby(['raceId','driverId']).size().reset_index(name='pit_stop_count')
    results = results.merge(pit_counts, on=['raceId','driverId'], how='left')
    results['pit_stop_count'] = results['pit_stop_count'].fillna(0)

    # Avg lap time (seconds)
    lap_times['lap_time_sec'] = lap_times['milliseconds'] / 1000.0
    avg_lap = lap_times.groupby(['raceId','driverId'])['lap_time_sec'].mean().reset_index(name='avg_lap_time')
    results = results.merge(avg_lap, on=['raceId','driverId'], how='left')

    # Rolling form features (last 3/5 races)
    for window in [3,5]:
        results = add_rolling_form(results, form_window=window, form_col=f'driver_form_last{window}', group_col='driverId')
        results = add_rolling_form(results, form_window=window, form_col=f'team_form_last{window}', group_col='constructorId')

    # Feature interactions
    results['grid_vs_qual'] = results['grid'] - results['position_qual']
    results['pit_lap_interaction'] = results['pit_stop_count'] * results['avg_lap_time']

    # Select and rename features
    results['driver_name'] = results['driver_forename'] + ' ' + results['driver_surname']
    
    features = [
        'raceId', 'year', 'circuit', 'driver_name', 'team_name',
        'grid', 'q1', 'q2', 'q3', 'pit_stop_count', 'avg_lap_time',
        'driver_form_last3', 'driver_form_last5', 'team_form_last3', 'team_form_last5',
        'grid_vs_qual', 'pit_lap_interaction', 'positionOrder'
    ]
    
    final_df = results[features].copy()

    # Handle invalid positionOrder values
    final_df = final_df[pd.to_numeric(final_df['positionOrder'], errors='coerce').notna()]
    final_df['positionOrder'] = final_df['positionOrder'].astype(int)

    return final_df

if __name__ == '__main__':
    print("Running feature engineering...")
    engineered_data = load_and_engineer_features()
    print("Feature engineering complete.")
    print("Engineered data shape:", engineered_data.shape)
    print("Engineered data columns:", engineered_data.columns.tolist())
    print(engineered_data.head())
    
    # Save the engineered data to a file for inspection
    output_path = 'data/engineered_features.csv'
    os.makedirs('data', exist_ok=True)
    engineered_data.to_csv(output_path, index=False)
    print(f"Engineered data saved to {output_path}")