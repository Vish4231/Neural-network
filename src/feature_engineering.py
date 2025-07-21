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

# --- Track Features: Expanded for Maximum Differentiation ---
# All keys are lowercased and use spaces
track_features = {
    'circuit de spa francorchamps': {
        'length_km': 7.004,
        'turns': 19,
        'elevation': 102,
        'drs_zones': 2,
        'grip': 7,
        'rain_prob': 0.6,
        'track_type': 'permanent',
        'overtaking_difficulty': 5,
        'pit_lane_time_loss': 22.0,
        'avg_lap_speed': 230,
        'surface_type': 'asphalt',
        'track_width': 10,
        'safety_car_prob': 0.4,
        'tyre_deg': 'medium',
        'corner_type_dist': {'slow': 6, 'medium': 8, 'fast': 5}
    },
    'yas marina circuit': {
        'length_km': 5.281,
        'turns': 16,
        'elevation': 5,
        'drs_zones': 3,
        'grip': 8,
        'rain_prob': 0.01,
        'track_type': 'permanent',
        'overtaking_difficulty': 3,
        'pit_lane_time_loss': 21.5,
        'avg_lap_speed': 210,
        'surface_type': 'asphalt',
        'track_width': 12,
        'safety_car_prob': 0.3,
        'tyre_deg': 'low',
        'corner_type_dist': {'slow': 7, 'medium': 6, 'fast': 3}
    },
    'circuit de monaco': {
        'length_km': 3.337,
        'turns': 19,
        'elevation': 42,
        'drs_zones': 1,
        'grip': 5,
        'rain_prob': 0.2,
        'track_type': 'street',
        'overtaking_difficulty': 1,
        'pit_lane_time_loss': 19.5,
        'avg_lap_speed': 160,
        'surface_type': 'asphalt',
        'track_width': 9,
        'safety_car_prob': 0.7,
        'tyre_deg': 'low',
        'corner_type_dist': {'slow': 12, 'medium': 5, 'fast': 2}
    },
    # ... Add all other circuits similarly ...
}

# --- Circuit Aliases: Map all common names to canonical keys ---
circuit_aliases = {
    # Spa
    'spa-francorchamps': 'circuit de spa francorchamps',
    'spa francorchamps': 'circuit de spa francorchamps',
    'spa': 'circuit de spa francorchamps',
    # Yas Marina
    'yas marina': 'yas marina circuit',
    'yas marina circuit': 'yas marina circuit',
    'abu dhabi': 'yas marina circuit',
    # Monaco
    'monaco': 'circuit de monaco',
    'circuit de monaco': 'circuit de monaco',
    'monte carlo': 'circuit de monaco',
    # Add more as needed for all tracks in your dataset
}

def normalize_circuit_name(name):
    original = name
    norm = name.strip().lower().replace('-', ' ').replace('_', ' ')
    mapped = circuit_aliases.get(norm, norm)
    print(f"[normalize_circuit_name] input: '{original}' | normalized: '{norm}' | mapped: '{mapped}'")
    return mapped

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

    # Add track-specific features
    track_type_map = {
        'Monaco': 'street', 'Baku': 'street', 'Singapore': 'street', 'Jeddah': 'street',
        'Silverstone': 'permanent', 'Spa-Francorchamps': 'permanent', 'Monza': 'permanent',
        'Hungaroring': 'permanent', 'Suzuka': 'permanent', 'Interlagos': 'permanent',
        # Add more circuits as needed
    }
    overtaking_map = {
        'Monaco': 1, 'Baku': 4, 'Singapore': 2, 'Jeddah': 5,
        'Silverstone': 4, 'Spa-Francorchamps': 5, 'Monza': 5, 'Hungaroring': 2, 'Suzuka': 3, 'Interlagos': 4,
        # Add more circuits as needed
    }
    results['track_type'] = results['circuit'].map(track_type_map).fillna('permanent')
    results['overtaking_difficulty'] = results['circuit'].map(overtaking_map).fillna(3)

    # --- Map all differentiating track features for each row ---
    differentiating_features = [
        'length_km', 'turns', 'elevation', 'drs_zones', 'grip', 'rain_prob', 'track_type',
        'overtaking_difficulty', 'pit_lane_time_loss', 'avg_lap_speed', 'surface_type',
        'track_width', 'safety_car_prob', 'tyre_deg', 'corner_type_dist'
    ]
    for feature in differentiating_features:
        results[feature] = results['circuit'].map(lambda x: track_features.get(normalize_circuit_name(x), {}).get(feature, np.nan))

    # --- Impute missing values for new features ---
    for col in ['length_km', 'turns', 'elevation', 'drs_zones', 'grip', 'rain_prob', 'overtaking_difficulty', 'pit_lane_time_loss', 'avg_lap_speed', 'track_width', 'safety_car_prob']:
        if col in results.columns:
            results[col] = results[col].fillna(results[col].median())
    for col in ['track_type', 'surface_type', 'tyre_deg']:
        if col in results.columns:
            mode = results[col].mode()[0] if not results[col].mode().empty else 'Unknown'
            results[col] = results[col].fillna(mode)
    if 'corner_type_dist' in results.columns:
        results['corner_type_dist'] = results['corner_type_dist'].apply(lambda x: x if isinstance(x, dict) else {'slow': 0, 'medium': 0, 'fast': 0})

    # --- Outlier Removal for Key Numeric Features ---
    # (Removed to preserve class balance and maximize accuracy)

    # --- Impute missing values for all features ---
    for col in results.select_dtypes(include=[np.number]).columns:
        results[col] = results[col].fillna(results[col].median())
    for col in results.select_dtypes(include=['object']).columns:
        mode = results[col].mode()[0] if not results[col].mode().empty else 'Unknown'
        results[col] = results[col].fillna(mode)

    # Print class distribution for Top 5 vs Not Top 5 after imputation
    if 'positionOrder' in results.columns:
        top5 = (results['positionOrder'] <= 5).astype(int)
        print("Class distribution after imputation (Top 5=1, Not Top 5=0):")
        print(top5.value_counts(normalize=True))

    # Select and rename features
    results['driver_name'] = results['driver_forename'] + ' ' + results['driver_surname']
    
    features = [
        'raceId', 'year', 'circuit', 'driver_name', 'team_name',
        'grid', 'q1', 'q2', 'q3', 'pit_stop_count', 'avg_lap_time',
        'driver_form_last3', 'driver_form_last5', 'team_form_last3', 'team_form_last5',
        'grid_vs_qual', 'pit_lap_interaction', 'track_type', 'overtaking_difficulty', 'positionOrder'
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