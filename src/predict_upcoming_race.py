import pandas as pd
import numpy as np
import os
import joblib
import argparse
import requests
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from tensorflow import keras
import sys

# Import the centralized feature engineering function
from feature_engineering import load_and_engineer_features, track_features, normalize_circuit_name

# --- Configuration ---
MODELS_DIR = 'model'
DATA_DIR = 'data'
F1_2025_DATASET_DIR = 'F1_2025_Dataset'

# --- Model and Preprocessor Loading ---
def load_prediction_artifacts():
    """Loads all trained models and preprocessing objects."""
    artifacts = {}
    try:
        artifacts['encoders'] = joblib.load(os.path.join(MODELS_DIR, 'encoders_top5.pkl'))
        artifacts['scaler'] = joblib.load(os.path.join(MODELS_DIR, 'scaler_top5.pkl'))
        artifacts['xgb_model'] = xgb.XGBClassifier()
        artifacts['xgb_model'].load_model(os.path.join(MODELS_DIR, 'xgb_top5.model'))
        artifacts['lgbm_model'] = lgb.Booster(model_file=os.path.join(MODELS_DIR, 'lgbm_top5.txt'))
        artifacts['cat_model'] = cb.CatBoostClassifier()
        artifacts['cat_model'].load_model(os.path.join(MODELS_DIR, 'catboost_top5.cbm'))
        artifacts['nn_model'] = keras.models.load_model(os.path.join(MODELS_DIR, 'pre_race_model_top5.keras'))
        artifacts['meta_model'] = joblib.load(os.path.join(MODELS_DIR, 'meta_model_logreg.pkl'))
        print("All prediction artifacts loaded successfully.")
    except FileNotFoundError as e:
        print(f"Error loading artifacts: {e}. Please run the training script first.")
        exit(1)
    return artifacts

# --- Data Fetching for Upcoming Race ---
def get_2025_lineup(circuit_name):
    """
    Loads the 2025 driver lineup and grid positions from the race results file for Silverstone,
    otherwise uses qualifying results for other circuits if available.
    """
    F1_2025_DATASET_DIR = 'F1_2025_Dataset'
    qual_path = os.path.join(F1_2025_DATASET_DIR, 'F1_2025_QualifyingResults.csv')
    race_results_path = os.path.join(F1_2025_DATASET_DIR, 'F1_2025_RaceResults.csv')

    # For Silverstone, always use race results file
    if circuit_name.lower() == "silverstone":
        df = pd.read_csv(race_results_path)
        race_df = df[df['Track'].str.lower() == circuit_name.lower()].copy()
        if race_df.empty:
            raise ValueError(f"No data found for circuit '{circuit_name}' in 2025 dataset.")
        lineup = race_df[['Driver', 'Team', 'Starting Grid']].rename(columns={
            'Driver': 'driver_name',
            'Team': 'team_name',
            'Starting Grid': 'grid'
        })
        return lineup

    # Otherwise, try to load qualifying results for the circuit
    if os.path.exists(qual_path):
        qual_df = pd.read_csv(qual_path)
        qual_circuit = qual_df[qual_df['Track'].str.lower() == circuit_name.lower()]
        # Only use numeric positions
        qual_circuit = qual_circuit[pd.to_numeric(qual_circuit['Position'], errors='coerce').notnull()]
        if not qual_circuit.empty:
            qual_circuit['Position'] = qual_circuit['Position'].astype(int)
            qual_circuit = qual_circuit.sort_values('Position')
            lineup = qual_circuit[['Driver', 'Team', 'Position']].rename(columns={
                'Driver': 'driver_name',
                'Team': 'team_name',
                'Position': 'grid'
            })
            return lineup

    # Fallback: use race results file
    df = pd.read_csv(race_results_path)
    race_df = df[df['Track'].str.lower() == circuit_name.lower()].copy()
    if race_df.empty:
        raise ValueError(f"No data found for circuit '{circuit_name}' in 2025 dataset.")
    lineup = race_df[['Driver', 'Team', 'Starting Grid']].rename(columns={
        'Driver': 'driver_name',
        'Team': 'team_name',
        'Starting Grid': 'grid'
    })
    return lineup

def standardize_2025_data(historical_df):
    """Loads 2025 data and standardizes it to match the historical format."""
    filepath = os.path.join(F1_2025_DATASET_DIR, 'F1_2025_RaceResults.csv')
    if not os.path.exists(filepath):
        return pd.DataFrame() # Return empty if no 2025 data exists

    df_2025 = pd.read_csv(filepath)
    
    standardized_rows = []
    last_race_id = historical_df['raceId'].max()
    
    # Assign a unique raceId to each race in the 2025 dataset
    race_mapping = {track: last_race_id + i + 1 for i, track in enumerate(df_2025['Track'].unique())}

    for _, row in df_2025.iterrows():
        new_row = {col: np.nan for col in historical_df.columns}
        new_row['raceId'] = race_mapping[row['Track']]
        new_row['year'] = 2025
        new_row['circuit'] = row['Track']
        new_row['driver_name'] = row['Driver']
        new_row['team_name'] = row['Team']
        new_row['grid'] = row['Starting Grid']
        new_row['positionOrder'] = row['Position']
        standardized_rows.append(new_row)
        
    df = pd.DataFrame(standardized_rows)
    # Clean the positionOrder column to handle non-numeric values like 'NC'
    df['positionOrder'] = pd.to_numeric(df['positionOrder'], errors='coerce')
    df = df.dropna(subset=['positionOrder'])
    df['positionOrder'] = df['positionOrder'].astype(int)
    
    return df

circuit_aliases = {
    'spa-francorchamps': 'circuit de spa-francorchamps',
    'spa': 'circuit de spa-francorchamps',
    'silverstone': 'silverstone circuit',
    'monaco': 'circuit de monaco',
    'baku': 'baku city circuit',
    'jeddah': 'jeddah street circuit',
    'imola': 'imola',
    'barcelona': 'circuit de barcelona',
    'montreal': 'circuit gilles villeneuve',
    'austria': 'red bull ring',
    'hungary': 'hungaroring',
    'zandvoort': 'circuit zandvoort',
    'monza': 'autodromo nazionale di monza',
    'singapore': 'marina bay street circuit',
    'suzuka': 'suzuka international racing course',
    'losail': 'losail international circuit',
    'cota': 'circuit of the americas',
    'mexico city': 'autódromo hermanos rodríguez',
    'interlagos': 'autódromo josé carlos pace',
    'las vegas': 'las vegas street circuit',
    'abu dhabi': 'yas marina circuit',
    # Add more aliases as needed
}

def normalize_circuit_name(name):
    norm = name.strip().lower().replace('-', ' ').replace('_', ' ')
    return circuit_aliases.get(norm, norm)

# --- Feature Generation for Prediction ---
def create_prediction_df(lineup, year, circuit, combined_df):
    print(f"[create_prediction_df] input circuit: '{circuit}'")
    canonical_circuit = normalize_circuit_name(circuit)  # This is the canonical key
    print(f"[create_prediction_df] canonical circuit: '{canonical_circuit}'")
    track_info = track_features.get(canonical_circuit, {})
    print(f"[create_prediction_df] track_info for '{canonical_circuit}': {track_info}\n")

    pred_rows = []
    for i, driver_info in lineup.iterrows():
        driver_name = driver_info['driver_name']
        team_name = driver_info['team_name']
        row = {
            'driver_name': driver_name,
            'team_name': team_name,
            'circuit': canonical_circuit,
            # ... other features ...
            **track_info
        }
        pred_rows.append(row)
    pred_df = pd.DataFrame(pred_rows)

    # Ensure all required track features are present
    required_features = [
        'length_km', 'turns', 'elevation', 'drs_zones', 'grip', 'rain_prob', 'track_type',
        'overtaking_difficulty', 'pit_lane_time_loss', 'avg_lap_speed', 'surface_type',
        'track_width', 'safety_car_prob', 'tyre_deg', 'corner_type_dist'
    ]
    for feat in required_features:
        if feat not in pred_df.columns:
            pred_df[feat] = np.nan if feat != 'track_type' and feat != 'surface_type' and feat != 'tyre_deg' and feat != 'corner_type_dist' else 'Unknown'
    print("[DEBUG] Prediction DataFrame (driver, team, circuit, track features):")
    print(pred_df[['driver_name', 'team_name', 'circuit'] + required_features])
    return pred_df

# --- Main Prediction Logic ---
def load_and_combine_data():
    try:
        # Load historical data
        historical_df = load_and_engineer_features()
        print(f"Loaded historical data. Shape: {historical_df.shape}")

        # Load and standardize 2025 data
        df_2025_std = standardize_2025_data(historical_df)
        print(f"Loaded and standardized 2025 data. Shape: {df_2025_std.shape}")

        # Combine the dataframes
        combined_df = pd.concat([historical_df, df_2025_std], ignore_index=True)
        print(f"Combined historical and 2025 data. Total records: {len(combined_df)}")

        return combined_df
    except Exception as e:
        print(f"Error in load_and_combine_data: {e}")
        print(f"Error details: {sys.exc_info()[2]}")
        return None

def get_2025_spa_lineup():
    import pandas as pd
    quali_path = 'F1_2025_Dataset/F1_2025_QualifyingResults.csv'
    race_path = 'F1_2025_Dataset/F1_2025_RaceResults.csv'
    # Qualifying lineup
    quali_df = pd.read_csv(quali_path)
    spa_quali = quali_df[quali_df['Track'] == 'Spa-Francorchamps']
    if spa_quali.empty:
        print('No Spa-Francorchamps entry found in 2025 qualifying results. Skipping quali simulation.')
        quali_lineup = None
    else:
        spa_quali = spa_quali[pd.to_numeric(spa_quali['Position'], errors='coerce').notnull()]
        spa_quali['Position'] = spa_quali['Position'].astype(int)
        spa_quali = spa_quali.sort_values('Position')
        if spa_quali.empty:
            print('No Spa qualifying data with valid positions. Skipping qualifying simulation.')
            quali_lineup = None
        else:
            quali_lineup = spa_quali[['Driver', 'Team', 'Position']].rename(columns={'Driver': 'driver_name', 'Team': 'team_name', 'Position': 'grid'})
    # Race lineup
    race_df = pd.read_csv(race_path)
    spa_race = race_df[race_df['Track'] == 'Spa-Francorchamps']
    spa_race = spa_race[pd.to_numeric(spa_race['Starting Grid'], errors='coerce').notnull()]
    spa_race['Starting Grid'] = spa_race['Starting Grid'].astype(int)
    spa_race = spa_race.sort_values('Starting Grid')
    race_lineup = spa_race[['Driver', 'Team', 'Starting Grid']].rename(columns={'Driver': 'driver_name', 'Team': 'team_name', 'Starting Grid': 'grid'})
    return quali_lineup, race_lineup

def predict_for_spa_2025():
    # Use up-to-date Spa 2025 lineups
    quali_lineup, race_lineup = get_2025_spa_lineup()
    from feature_engineering import engineer_features_for_prediction, track_features
    # Load combined data for feature engineering
    combined_df = load_and_combine_data()
    import joblib
    features = ['driver_skill', 'driver_form_last3', 'team_form_last3', 'length_km', 'turns', 'elevation', 'drs_zones', 'grip', 'rain_prob']
    cat_features = ['driver_name', 'team_name']
    # --- Qualifying Prediction ---
    if quali_lineup is not None and not quali_lineup.empty:
        quali_pred_df = engineer_features_for_prediction(quali_lineup, combined_df)
        if quali_pred_df.empty:
            print('No Spa qualifying data after feature engineering. Skipping qualifying simulation.')
        else:
            qual_model = joblib.load('model/qualifying_rf_model.pkl')
            qual_encoders = joblib.load('model/qualifying_encoders.pkl')
            qual_scaler = joblib.load('model/qualifying_scaler.pkl')
            X = quali_pred_df[features + cat_features].fillna(0)
            for col in cat_features:
                le = qual_encoders[col]
                X[col] = le.transform(X[col].astype(str))
            X[features] = qual_scaler.transform(X[features])
            quali_preds = qual_model.predict(X)
            quali_lineup = quali_lineup.iloc[:len(quali_preds)].copy()
            quali_lineup['predicted_grid'] = quali_preds
            print('\n--- Predicted Spa 2025 Qualifying Top 5 ---')
            for i, row in quali_lineup.sort_values('predicted_grid').head(5).iterrows():
                print(f"{int(row['predicted_grid'])}. {row['driver_name']} ({row['team_name']})")
    else:
        print('No Spa qualifying data available. Skipping qualifying simulation.')
    # --- Race Prediction ---
    race_pred_df = engineer_features_for_prediction(race_lineup, combined_df)
    race_model = joblib.load('model/race_rf_model.pkl')
    race_encoders = joblib.load('model/race_encoders.pkl')
    race_scaler = joblib.load('model/race_scaler.pkl')
    X_race = race_pred_df[features + cat_features].fillna(0)
    for col in cat_features:
        le = race_encoders[col]
        X_race[col] = le.transform(X_race[col].astype(str))
    X_race[features] = race_scaler.transform(X_race[features])
    race_probs = race_model.predict_proba(X_race)[:, 1]
    race_lineup['race_top5_prob'] = race_probs
    print('\n--- Predicted Spa 2025 Race Top 5 Probabilities ---')
    for i, row in race_lineup.sort_values('race_top5_prob', ascending=False).head(5).iterrows():
        print(f"{row['driver_name']} ({row['team_name']}) - Top 5 Probability: {row['race_top5_prob']:.2%}")

import pandas as pd
import joblib
from feature_engineering import engineer_f1db_features, track_features

def predict_for_spa_2025():
    # Load merged and engineered data
    df = pd.read_csv('data/f1db_merged_2010_2025.csv')
    df = engineer_f1db_features(df, track_features)
    # Filter for Spa 2025 drivers/teams (simulate upcoming race)
    spa_circuit_key = 'circuit de spa-francorchamps'
    latest_year = 2025
    # Get the latest known lineup for Spa
    spa_lineup = df[(df['year'] == latest_year) & (df['circuit_key'] == spa_circuit_key)]
    # If not available, fallback to 2024
    if spa_lineup.empty:
        spa_lineup = df[(df['year'] == 2024) & (df['circuit_key'] == spa_circuit_key)]
    # Use only the latest entry per driver
    spa_lineup = spa_lineup.sort_values('date').groupby('driverId').tail(1)
    features = ['driver_skill', 'driver_form_last3', 'team_form_last3', 'length_km', 'turns', 'elevation', 'drs_zones', 'grip', 'rain_prob']
    cat_features = ['driverId', 'constructorId']
    X = spa_lineup[features + cat_features].fillna(0)
    # --- Qualifying Prediction ---
    qual_model = joblib.load('model/qualifying_rf_model.pkl')
    qual_encoders = joblib.load('model/qualifying_encoders.pkl')
    qual_scaler = joblib.load('model/qualifying_scaler.pkl')
    for col in cat_features:
        le = qual_encoders[col]
        X[col] = le.transform(X[col].astype(str))
    X[features] = qual_scaler.transform(X[features])
    quali_preds = qual_model.predict(X)
    spa_lineup['predicted_grid'] = quali_preds
    print('\n--- Predicted Spa 2025 Qualifying Top 5 ---')
    for i, row in spa_lineup.sort_values('predicted_grid').head(5).iterrows():
        print(f"{int(row['predicted_grid'])}. {row['fullName']} ({row['name_constructor']})")
    # --- Race Prediction ---
    race_model = joblib.load('model/race_rf_model.pkl')
    race_encoders = joblib.load('model/race_encoders.pkl')
    race_scaler = joblib.load('model/race_scaler.pkl')
    X_race = spa_lineup[features + cat_features].fillna(0)
    for col in cat_features:
        le = race_encoders[col]
        X_race[col] = le.transform(X_race[col].astype(str))
    X_race[features] = race_scaler.transform(X_race[features])
    race_probs = race_model.predict_proba(X_race)[:, 1]
    spa_lineup['race_top5_prob'] = race_probs
    print('\n--- Predicted Spa 2025 Race Top 5 Probabilities ---')
    for i, row in spa_lineup.sort_values('race_top5_prob', ascending=False).head(5).iterrows():
        print(f"{row['fullName']} ({row['name_constructor']}) - Top 5 Probability: {row['race_top5_prob']:.2%}")

if __name__ == '__main__':
    predict_for_spa_2025()

def get_2025_lineup_from_csv(track_name):
    import pandas as pd
    try:
        df = pd.read_csv('data/F1_2025_RaceResults.csv')
    except FileNotFoundError:
        print('F1_2025_RaceResults.csv not found.')
        return None
    # Filter for the given track
    lineup = df[df['Track'].str.lower() == track_name.lower()]
    if lineup.empty:
        print(f'No 2025 lineup found for {track_name}.')
        return None
    # Only use drivers with a valid grid position (exclude NC, DQ, etc.)
    lineup = lineup[lineup['Starting Grid'].apply(lambda x: str(x).isdigit())]
    lineup = lineup.sort_values('Starting Grid')
    return lineup[['Driver', 'Team', 'Starting Grid']]

def predict_for_yas_marina_2025():
    # Try to get the 2025 lineup from the CSV using the correct track name
    yas_csv_track_name = 'Abu Dhabi'  # This matches the value in F1_2025_RaceResults.csv
    yas_lineup = get_2025_lineup_from_csv(yas_csv_track_name)
    if yas_lineup is not None and not yas_lineup.empty:
        # Build a DataFrame with the required features for prediction
        # Use the most recent historical data for each driver/team to fill in features
        df_hist = pd.read_csv('data/f1db_merged_2010_2025.csv')
        df_hist = engineer_f1db_features(df_hist, track_features)
        features = ['driver_skill', 'driver_form_last3', 'team_form_last3', 'length_km', 'turns', 'elevation', 'drs_zones', 'grip', 'rain_prob']
        cat_features = ['driverId', 'constructorId']
        # Map driver/team to their most recent stats
        pred_rows = []
        for _, row in yas_lineup.iterrows():
            # Find the most recent entry for this driver/team at Yas Marina or elsewhere
            hist_row = df_hist[df_hist['fullName'].str.lower() == row['Driver'].lower()]
            if hist_row.empty:
                continue
            hist_row = hist_row.sort_values('date').iloc[-1]
            pred_row = {f: hist_row.get(f, 0) for f in features + cat_features}
            pred_row['fullName'] = row['Driver']
            pred_row['name_constructor'] = row['Team']
            pred_rows.append(pred_row)
        yas_lineup_df = pd.DataFrame(pred_rows)
        if yas_lineup_df.empty:
            print('No matching historical data for 2025 lineup.')
            return
        X = yas_lineup_df[features + cat_features].fillna(0)
    else:
        # Fallback to historical method
        df = pd.read_csv('data/f1db_merged_2010_2025.csv')
        df = engineer_f1db_features(df, track_features)
        yas_circuit_key = 'Yas Marina'
        for year in [2025, 2024, 2023, 2022, 2021]:
            yas_lineup_df = df[(df['year'] == year) & (df['name_circuit'] == yas_circuit_key)]
            if not yas_lineup_df.empty:
                break
        if yas_lineup_df.empty:
            print('No Yas Marina lineup found for recent years.')
            return
        yas_lineup_df = yas_lineup_df.sort_values('date').groupby('driverId').tail(1)
        features = ['driver_skill', 'driver_form_last3', 'team_form_last3', 'length_km', 'turns', 'elevation', 'drs_zones', 'grip', 'rain_prob']
        cat_features = ['driverId', 'constructorId']
        X = yas_lineup_df[features + cat_features].fillna(0)
    # --- Qualifying Prediction ---
    qual_model = joblib.load('model/qualifying_rf_model.pkl')
    qual_encoders = joblib.load('model/qualifying_encoders.pkl')
    qual_scaler = joblib.load('model/qualifying_scaler.pkl')
    for col in cat_features:
        le = qual_encoders[col]
        X[col] = le.transform(X[col].astype(str))
    X[features] = qual_scaler.transform(X[features])
    quali_preds = qual_model.predict(X)
    yas_lineup_df['predicted_grid'] = quali_preds
    print('\n--- Predicted Yas Marina 2025 Qualifying Top 5 ---')
    for i, row in yas_lineup_df.sort_values('predicted_grid').head(5).iterrows():
        print(f"{int(row['predicted_grid'])}. {row['fullName']} ({row['name_constructor']})")
    # --- Race Prediction ---
    race_model = joblib.load('model/race_rf_model.pkl')
    race_encoders = joblib.load('model/race_encoders.pkl')
    race_scaler = joblib.load('model/race_scaler.pkl')
    X_race = yas_lineup_df[features + cat_features].fillna(0)
    for col in cat_features:
        le = race_encoders[col]
        X_race[col] = le.transform(X_race[col].astype(str))
    X_race[features] = race_scaler.transform(X_race[features])
    race_probs = race_model.predict_proba(X_race)[:, 1]
    yas_lineup_df['race_top5_prob'] = race_probs
    print('\n--- Predicted Yas Marina 2025 Race Top 5 Probabilities ---')
    for i, row in yas_lineup_df.sort_values('race_top5_prob', ascending=False).head(5).iterrows():
        print(f"{row['fullName']} ({row['name_constructor']}) - Top 5 Probability: {row['race_top5_prob']:.2%}")

# To run Yas Marina prediction, uncomment below:
# if __name__ == '__main__':
#     predict_for_yas_marina_2025()