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
    norm_circuit = normalize_circuit_name(circuit)
    track_info = track_features.get(norm_circuit, {})
    track_type = track_info.get('track_type', 'permanent')
    overtaking_difficulty = track_info.get('overtaking_difficulty', 3)

    pred_rows = []
    for i, driver_info in lineup.iterrows():
        driver_name = driver_info['driver_name']
        team_name = driver_info['team_name']
        driver_hist = combined_df[combined_df['driver_name'] == driver_name].copy()
        driver_hist = driver_hist.sort_values('raceId')
        team_hist = combined_df[combined_df['team_name'] == team_name].copy()
        team_hist = team_hist.sort_values('raceId')
        row = {
            'year': year,
            'circuit': circuit,
            'driver_name': driver_name,
            'team_name': team_name,
            'grid': driver_info['grid'],
            'q1': driver_hist['q1'].median(),
            'q2': driver_hist['q2'].median(),
            'q3': driver_hist['q3'].median(),
            'pit_stop_count': driver_hist['pit_stop_count'].median(),
            'avg_lap_time': driver_hist['avg_lap_time'].median(),
            'driver_form_last3': driver_hist['positionOrder'].rolling(3, min_periods=1).mean().iloc[-1] if not driver_hist.empty else 10,
            'driver_form_last5': driver_hist['positionOrder'].rolling(5, min_periods=1).mean().iloc[-1] if not driver_hist.empty else 10,
            'team_form_last3': team_hist['positionOrder'].rolling(3, min_periods=1).mean().iloc[-1] if not team_hist.empty else 10,
            'team_form_last5': team_hist['positionOrder'].rolling(5, min_periods=1).mean().iloc[-1] if not team_hist.empty else 10,
            'grid_vs_qual': 0,
            'pit_lap_interaction': 0,
            'track_type': track_type,
            'overtaking_difficulty': overtaking_difficulty
        }
        pred_rows.append(row)
    pred_df = pd.DataFrame(pred_rows)
    features = [col for col in combined_df.columns if col not in ['positionOrder', 'raceId']]
    return pred_df[features]

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

def main():
    print("--- Predicting Top 5 for Spa-Francorchamps 2025 ---")

    # Load models and preprocessors
    artifacts = load_prediction_artifacts()
    print("All prediction artifacts loaded successfully.")

    # Load and combine data
    combined_df = load_and_combine_data()
    if combined_df is None:
        print("Failed to load and combine data. Exiting.")
        return

    # Get lineup for the upcoming race
    try:
        lineup = get_2025_lineup("spa-francorchamps")
        print(f"\nLoaded lineup for {len(lineup)} drivers.")
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        return

    # Create the feature set for the prediction using the combined data
    pred_df = create_prediction_df(lineup, 2025, "spa-francorchamps", combined_df)

    # Print prediction features for debugging
    print("\nPrediction features for this race:")
    print(pred_df)

    # Preprocess the prediction data
    pred_processed = pred_df.copy()
    
    # Define feature lists from the combined dataframe
    features = [col for col in combined_df.columns if col not in ['positionOrder', 'raceId']]
    cat_features = ['team_name', 'driver_name', 'circuit', 'track_type']
    num_features = [f for f in features if f not in cat_features and f != 'year']

    # Impute any missing values that might have occurred
    for col in num_features:
        if col in pred_processed.columns:
            pred_processed[col] = pred_processed[col].fillna(combined_df[col].median())
    for col in cat_features:
        if col in pred_processed.columns:
            mode_series = combined_df[col].mode()
            if not mode_series.empty:
                mode = mode_series[0]
            else:
                mode = "Unknown"
            pred_processed[col] = pred_processed[col].fillna(mode)

    # Encode categorical features
    for col in cat_features:
        le = artifacts['encoders'][col]
        # Handle unseen labels by mapping them to a known class
        pred_processed[col] = pred_processed[col].astype(str).apply(lambda x: x if x in le.classes_ else le.classes_[0])
        pred_processed[col] = le.transform(pred_processed[col])

    # Scale numerical features
    pred_processed.loc[:, num_features] = artifacts['scaler'].transform(pred_processed[num_features])
    print("Prediction data preprocessed successfully.")

    # Make predictions with all models
    print("\nGenerating predictions from all models...")
    X_pred = pred_processed[features]
    
    xgb_probs = artifacts['xgb_model'].predict_proba(X_pred)[:, 1]
    lgbm_probs = artifacts['lgbm_model'].predict(X_pred)
    cat_probs = artifacts['cat_model'].predict_proba(X_pred)[:, 1]
    nn_probs = artifacts['nn_model'].predict(X_pred).flatten()

    # Use the meta-model for the final ensemble prediction
    stack_X = np.vstack([xgb_probs, lgbm_probs, cat_probs, nn_probs]).T
    ensemble_probs = artifacts['meta_model'].predict_proba(stack_X)[:, 1]

    # Display results
    results_df = lineup.copy()
    results_df['probability'] = ensemble_probs
    results_df = results_df.sort_values('probability', ascending=False).reset_index(drop=True)

    print("\n--- Predicted Top 5 Finishers (Ensemble Model) ---")
    for i, row in results_df.head(5).iterrows():
        medal = ["🥇", "🥈", "🥉", "🏅", "🏅"][i]
        print(f"{i+1}. {medal} {row['driver_name']} ({row['team_name']}) - Probability: {row['probability']:.2%}")

    print("\n--- Full Probability Ranking ---")
    print(results_df.to_string(index=False))

if __name__ == "__main__":
    main()