#!/usr/bin/env python3
"""
Simplified F1 Race Prediction Script
Skips actual results to avoid API rate limiting issues
"""

import requests
import pandas as pd
import numpy as np
import os
import joblib
from tensorflow import keras
import argparse
import time
from typing import Optional

def main():
    parser = argparse.ArgumentParser(description="Predict F1 race results.")
    parser.add_argument('--year', type=int, default=2025, help='Year of the race (e.g., 2025)')
    parser.add_argument('--circuit', type=str, required=True, help='Circuit short name (e.g., Melbourne)')
    args = parser.parse_args()
    
    YEAR = args.year
    CIRCUIT = args.circuit
    
    # Load models
    try:
        model = keras.models.load_model('model/pre_race_model_top5.keras')
        encoders = joblib.load('model/encoders_top5.pkl')
        scaler = joblib.load('model/scaler_top5.pkl')
        
        import xgboost as xgb
        xgb_model = xgb.XGBClassifier()
        xgb_model.load_model('model/xgb_top5.model')
        
        print(f"✅ Models loaded successfully")
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return
    
    # Helper function with basic retry
    def api_get(url, max_retries=2):
        for attempt in range(max_retries):
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:
                    print(f"Rate limited, waiting {2 ** attempt} seconds...")
                    time.sleep(2 ** attempt)
                else:
                    print(f"API error: {response.status_code}")
                    return None
            except Exception as e:
                print(f"Request error: {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)
        return None
    
    # Get race session
    print(f"🏁 Looking for {CIRCUIT} {YEAR} race session...")
    sessions_url = f"https://api.openf1.org/v1/sessions?session_type=Race&year={YEAR}"
    sessions = api_get(sessions_url)
    
    if not sessions:
        print("❌ Could not fetch sessions")
        return
    
    # Find the circuit
    session = None
    for s in sessions:
        if str(s.get('circuit_short_name', '')).lower() == CIRCUIT.lower():
            session = s
            break
    
    if not session:
        available = sorted(set(s.get('circuit_short_name','') for s in sessions))
        print(f"❌ Circuit '{CIRCUIT}' not found for {YEAR}. Available circuits:")
        for c in available:
            print(f"  - {c}")
        return
    
    session_key = session['session_key']
    meeting_key = session['meeting_key']
    circuit_name = session.get('circuit_short_name', 'Unknown')
    country = session.get('country_name', 'Unknown')
    
    print(f"🏎️  Predicting for: {circuit_name} | {country} | {YEAR}")
    
    # Get basic race data
    print("📊 Fetching race data...")
    
    # Starting grid
    grid_url = f"https://api.openf1.org/v1/starting_grid?session_key={session_key}"
    grid = api_get(grid_url) or []
    
    # Driver info
    drivers_url = f"https://api.openf1.org/v1/drivers?session_key={session_key}"
    drivers = api_get(drivers_url) or []
    driver_map = {d['driver_number']: d for d in drivers}
    
    # Weather data
    weather_url = f"https://api.openf1.org/v1/weather?meeting_key={meeting_key}&session_key={session_key}"
    weather_data = api_get(weather_url) or []
    
    # Process weather
    weather = {}
    if weather_data:
        df_weather = pd.DataFrame(weather_data)
        weather = {
            'air_temperature': df_weather['air_temperature'].mean() if 'air_temperature' in df_weather else 20,
            'humidity': df_weather['humidity'].mean() if 'humidity' in df_weather else 50,
            'rainfall': df_weather['rainfall'].mean() if 'rainfall' in df_weather else 0,
            'track_temperature': df_weather['track_temperature'].mean() if 'track_temperature' in df_weather else 25,
            'wind_speed': df_weather['wind_speed'].mean() if 'wind_speed' in df_weather else 5,
        }
    else:
        # Default weather values
        weather = {
            'air_temperature': 20,
            'humidity': 50,
            'rainfall': 0,
            'track_temperature': 25,
            'wind_speed': 5,
        }
    
    print(f"🌤️  Weather: {weather['air_temperature']:.1f}°C, {weather['humidity']:.1f}% humidity, {weather['rainfall']:.1f}mm rain")
    
    # Build feature matrix
    rows = []
    for entry in grid:
        drv_num = entry['driver_number']
        drv = driver_map.get(drv_num, {})
        
        row = {
            'grid_position': entry['position'],
            'qualifying_lap_time': entry.get('lap_duration', 90.0),  # Default lap time
            'air_temperature': weather['air_temperature'],
            'humidity': weather['humidity'],
            'rainfall': weather['rainfall'],
            'track_temperature': weather['track_temperature'],
            'wind_speed': weather['wind_speed'],
            'team_name': drv.get('team_name', 'Unknown'),
            'driver_name': drv.get('full_name', f"Driver {drv_num}"),
            'circuit': circuit_name,
            'country_code': drv.get('country_code', 'UNK'),
            'driver_form_last3': 8.0,  # Default form
            'team_form_last3': 8.0,    # Default form
            'qualifying_gap_to_pole': 0.0,
            'teammate_grid_delta': 0.0,
            'track_type': 'permanent',
            'overtaking_difficulty': 3.0,
            'driver_championship_position': 0.0,
            'team_championship_position': 0.0,
            'driver_points_season': 0.0,
            'team_points_season': 0.0,
        }
        rows.append(row)
    
    if not rows:
        print("❌ No grid data available")
        return
    
    df = pd.DataFrame(rows)
    
    # Prepare features
    features = [
        'grid_position', 'qualifying_lap_time', 'air_temperature', 'humidity', 'rainfall',
        'track_temperature', 'wind_speed', 'team_name', 'driver_name', 'circuit', 'country_code',
        'driver_form_last3', 'team_form_last3', 'qualifying_gap_to_pole', 'teammate_grid_delta',
        'track_type', 'overtaking_difficulty',
        'driver_championship_position', 'team_championship_position', 'driver_points_season', 'team_points_season'
    ]
    
    # Encode categorical features
    cat_features = ['team_name', 'driver_name', 'circuit', 'country_code', 'track_type']
    for col in cat_features:
        if col in encoders:
            le = encoders[col]
            # Handle unseen labels by mapping to first class
            df[col] = df[col].astype(str).apply(lambda x: x if x in le.classes_ else le.classes_[0])
            df[col] = le.transform(df[col])
    
    # Scale numerical features
    num_features = [f for f in features if f not in cat_features]
    df[num_features] = scaler.transform(df[num_features])
    
    # Fill any remaining NaNs
    df[features] = df[features].fillna(0)
    
    print("🤖 Running predictions...")
    
    # Make predictions
    nn_probs = model.predict(df[features], verbose=0).flatten()
    xgb_probs = xgb_model.predict_proba(df[features])[:,1]
    ensemble_probs = (nn_probs + xgb_probs) / 2
    
    # Prepare results
    results_df = pd.DataFrame({
        'driver': [driver_map.get(g['driver_number'], {}).get('full_name', f"Driver {g['driver_number']}") for g in grid],
        'team': [driver_map.get(g['driver_number'], {}).get('team_name', 'Unknown') for g in grid],
        'grid_position': [g['position'] for g in grid],
        'top5_probability_nn': nn_probs,
        'top5_probability_xgb': xgb_probs,
        'top5_probability_ensemble': ensemble_probs
    })
    
    results_df = results_df.sort_values('top5_probability_ensemble', ascending=False).reset_index(drop=True)
    
    print(f"\n🏆 Top 5 Predicted Finishers - {circuit_name} {YEAR}:")
    print("=" * 60)
    
    medals = ["🥇", "🥈", "🥉", "🏅", "🏅"]
    for idx, row in results_df.head(5).iterrows():
        medal = medals[idx] if idx < 5 else ""
        print(f"{idx+1}. {medal} {row['driver']} ({row['team']})")
        print(f"   Grid: {row['grid_position']} | Ensemble Probability: {row['top5_probability_ensemble']:.1%}")
        print()
    
    print("📊 Full Prediction Results:")
    print("=" * 60)
    print(results_df.to_string(index=False, float_format='%.3f'))
    
    print(f"\n🎯 Neural Network vs XGBoost Comparison:")
    print("=" * 60)
    comparison = results_df[['driver', 'team', 'top5_probability_nn', 'top5_probability_xgb', 'top5_probability_ensemble']].head(10)
    print(comparison.to_string(index=False, float_format='%.3f'))
    
    print(f"\n✅ Prediction complete for {circuit_name} {YEAR}!")
    print("Note: This is a prediction based on historical data and current season form.")

if __name__ == "__main__":
    main()
