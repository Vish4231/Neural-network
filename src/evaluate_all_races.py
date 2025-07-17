import pandas as pd
import numpy as np
import os
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import xgboost as xgb

DATA_PATH = 'data/pre_race_features.csv'
MODEL_PATH = 'model/pre_race_model_top5.keras'
XGB_PATH = 'model/xgb_top5.model'
ENCODERS_PATH = 'model/encoders_top5.pkl'
SCALER_PATH = 'model/scaler_top5.pkl'

features = [
    'grid_position', 'qualifying_lap_time', 'air_temperature', 'humidity', 'rainfall',
    'track_temperature', 'wind_speed', 'team_name', 'driver_name', 'circuit', 'country_code',
    'driver_form_last3', 'team_form_last3', 'qualifying_gap_to_pole', 'teammate_grid_delta',
    'track_type', 'overtaking_difficulty',
    'driver_championship_position', 'team_championship_position', 'driver_points_season', 'team_points_season'
]

# Load data
all_df = pd.read_csv(DATA_PATH)
all_df = all_df.sort_values(['year', 'circuit', 'grid_position'])

# Load models and encoders
model = keras.models.load_model(MODEL_PATH)
xgb_model = xgb.XGBClassifier()
xgb_model.load_model(XGB_PATH)
encoders = joblib.load(ENCODERS_PATH)
scaler = joblib.load(SCALER_PATH)

results = []
race_keys = all_df[['year', 'circuit']].drop_duplicates().values.tolist()

for year, circuit in race_keys:
    race_df = all_df[(all_df['year'] == year) & (all_df['circuit'] == circuit)].copy()
    # Ensure all required features are present
    for col in ['driver_championship_position', 'team_championship_position', 'driver_points_season', 'team_points_season']:
        if col not in race_df.columns:
            race_df[col] = -1
    # Fill NaNs
    race_df[features] = race_df[features].fillna(-1)
    # Encode categoricals
    for col in ['team_name', 'driver_name', 'circuit', 'country_code', 'track_type']:
        le = encoders[col]
        race_df[col] = race_df[col].astype(str).apply(lambda x: x if x in le.classes_ else le.classes_[0])
        race_df[col] = le.transform(race_df[col])
    num_features = [f for f in features if f not in encoders]
    race_df[num_features] = scaler.transform(race_df[num_features])
    # Predict
    nn_probs = model.predict(race_df[features]).flatten()
    xgb_probs = xgb_model.predict_proba(race_df[features])[:,1]
    ensemble_probs = (nn_probs + xgb_probs) / 2
    race_df['top5_prob_nn'] = nn_probs
    race_df['top5_prob_xgb'] = xgb_probs
    race_df['top5_prob_ensemble'] = ensemble_probs
    # Get predicted top 5 (ensemble)
    pred_top5_ensemble = race_df.sort_values('top5_prob_ensemble', ascending=False).head(5)['driver_name'].tolist()
    pred_top5_nn = race_df.sort_values('top5_prob_nn', ascending=False).head(5)['driver_name'].tolist()
    pred_top5_xgb = race_df.sort_values('top5_prob_xgb', ascending=False).head(5)['driver_name'].tolist()
    # Get actual top 5
    actual_top5 = race_df.sort_values('finishing_position').head(5)['driver_name'].tolist()
    # Calculate intersection
    correct_ensemble = len(set(pred_top5_ensemble) & set(actual_top5))
    correct_nn = len(set(pred_top5_nn) & set(actual_top5))
    correct_xgb = len(set(pred_top5_xgb) & set(actual_top5))
    results.append({
        'year': year,
        'circuit': circuit,
        'correct_top5_ensemble': correct_ensemble,
        'correct_top5_nn': correct_nn,
        'correct_top5_xgb': correct_xgb,
        'pred_top5_ensemble': pred_top5_ensemble,
        'pred_top5_nn': pred_top5_nn,
        'pred_top5_xgb': pred_top5_xgb,
        'actual_top5': actual_top5
    })

# Summary
results_df = pd.DataFrame(results)
mean_correct_ensemble = results_df['correct_top5_ensemble'].mean()
mean_correct_nn = results_df['correct_top5_nn'].mean()
mean_correct_xgb = results_df['correct_top5_xgb'].mean()
print(f"\nMean correct top 5 per race (Ensemble): {mean_correct_ensemble:.2f}")
print(f"Mean correct top 5 per race (Neural Net): {mean_correct_nn:.2f}")
print(f"Mean correct top 5 per race (XGBoost): {mean_correct_xgb:.2f}")
print(f"Best race (Ensemble): {results_df['correct_top5_ensemble'].max()} correct; Worst race: {results_df['correct_top5_ensemble'].min()} correct")
print("\nRaces with perfect prediction (Ensemble):")
print(results_df[results_df['correct_top5_ensemble'] == 5][['year', 'circuit']])
print("\nRaces with 0 correct (Ensemble):")
print(results_df[results_df['correct_top5_ensemble'] == 0][['year', 'circuit']])
results_df.to_csv('model/eval_top5_results_ensemble.csv', index=False)
print("\nFull evaluation saved to model/eval_top5_results_ensemble.csv") 