import pandas as pd
import numpy as np
import os
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

DATA_PATH = 'data/pre_race_features.csv'
MODEL_PATH = 'model/pre_race_model_top5.keras'
ENCODERS_PATH = 'model/encoders_top5.pkl'
SCALER_PATH = 'model/scaler_top5.pkl'

features = [
    'grid_position', 'qualifying_lap_time', 'air_temperature', 'humidity', 'rainfall',
    'track_temperature', 'wind_speed', 'team_name', 'driver_name', 'circuit', 'country_code',
    'driver_form_last3', 'team_form_last3', 'qualifying_gap_to_pole', 'teammate_grid_delta',
    'track_type', 'overtaking_difficulty'
]

# Load data
all_df = pd.read_csv(DATA_PATH)
all_df = all_df.sort_values(['year', 'circuit', 'grid_position'])

# Load model and encoders
model = keras.models.load_model(MODEL_PATH)
encoders = joblib.load(ENCODERS_PATH)
scaler = joblib.load(SCALER_PATH)

results = []
race_keys = all_df[['year', 'circuit']].drop_duplicates().values.tolist()

for year, circuit in race_keys:
    race_df = all_df[(all_df['year'] == year) & (all_df['circuit'] == circuit)].copy()
    # Simulate pre-race: use only data up to this race for encoding/scaling
    # (for simplicity, use global encoders/scaler; for strictest evaluation, re-fit up to this race)
    for col in ['team_name', 'driver_name', 'circuit', 'country_code', 'track_type']:
        le = encoders[col]
        race_df[col] = race_df[col].astype(str).apply(lambda x: x if x in le.classes_ else le.classes_[0])
        race_df[col] = le.transform(race_df[col])
    num_features = [f for f in features if f not in encoders]
    race_df[num_features] = scaler.transform(race_df[num_features])
    # Predict
    probs = model.predict(race_df[features]).flatten()
    race_df['top5_prob'] = probs
    # Get predicted top 5
    pred_top5 = race_df.sort_values('top5_prob', ascending=False).head(5)['driver_name'].tolist()
    # Get actual top 5
    actual_top5 = race_df.sort_values('finishing_position').head(5)['driver_name'].tolist()
    # Calculate intersection
    correct = len(set(pred_top5) & set(actual_top5))
    results.append({
        'year': year,
        'circuit': circuit,
        'correct_top5': correct,
        'pred_top5': pred_top5,
        'actual_top5': actual_top5
    })

# Summary
results_df = pd.DataFrame(results)
mean_correct = results_df['correct_top5'].mean()
print(f"\nMean correct top 5 per race: {mean_correct:.2f}")
print(f"Best race: {results_df['correct_top5'].max()} correct; Worst race: {results_df['correct_top5'].min()} correct")
print("\nRaces with perfect prediction:")
print(results_df[results_df['correct_top5'] == 5][['year', 'circuit']])
print("\nRaces with 0 correct:")
print(results_df[results_df['correct_top5'] == 0][['year', 'circuit']])
results_df.to_csv('model/eval_top5_results.csv', index=False)
print("\nFull evaluation saved to model/eval_top5_results.csv") 