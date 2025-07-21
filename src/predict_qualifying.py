import pandas as pd
import numpy as np
import os
import argparse
import joblib
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder, StandardScaler
from feature_engineering import normalize_circuit_name, track_features

def load_qualifying_data(year, circuit):
    qual_path = 'archive (1)/qualifying.csv'
    races = pd.read_csv('archive (1)/races.csv')
    drivers = pd.read_csv('archive (1)/drivers.csv')
    constructors = pd.read_csv('archive (1)/constructors.csv')
    qual = pd.read_csv(qual_path)
    qual = qual.merge(races[['raceId','year','name']], on='raceId', how='left')
    qual = qual.merge(drivers[['driverId','surname','forename']], on='driverId', how='left')
    qual = qual.merge(constructors[['constructorId','name']], on='constructorId', how='left', suffixes=('', '_team'))
    qual = qual.rename(columns={'name':'circuit', 'name_team':'team_name', 'surname':'driver_surname', 'forename':'driver_forename'})
    qual = qual[(qual['year'] == year) & (qual['circuit'].str.lower() == circuit.lower())].copy()
    qual['driver_name'] = qual['driver_forename'] + ' ' + qual['driver_surname']
    for col in ['q1','q2','q3']:
        qual[col] = qual[col].replace('\\N', np.nan)
        qual[col] = qual[col].apply(lambda t: float(t.split(':')[0])*60+float(t.split(':')[1]) if isinstance(t, str) and ':' in t else (float(t) if pd.notnull(t) else np.nan))
    qual['best_quali_time'] = qual[['q1','q2','q3']].min(axis=1)
    qual = qual.sort_values(['driverId','year','raceId'])
    qual['qual_form_last3'] = qual.groupby('driverId')['position'].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    return qual

def generate_synthetic_qualifying(year, circuit):
    # Use 2025 race lineup for the requested track
    race_results_path = 'F1_2025_Dataset/F1_2025_RaceResults.csv'
    if not os.path.exists(race_results_path):
        return pd.DataFrame()
    race_df = pd.read_csv(race_results_path)
    lineup = race_df[race_df['Track'].str.lower() == circuit.lower()].copy()
    if lineup.empty:
        return pd.DataFrame()
    # Build synthetic qualifying entry
    df = pd.DataFrame({
        'year': year,
        'circuit': circuit,
        'driver_name': lineup['Driver'],
        'team_name': lineup['Team'],
        'best_quali_time': np.nan,
        'qual_form_last3': np.nan
    })
    # Save for pretty printing
    df['driver_name_orig'] = lineup['Driver']
    df['team_name_orig'] = lineup['Team']
    return df

def main():
    parser = argparse.ArgumentParser(description="Predict F1 qualifying order for a given track and year.")
    parser.add_argument('--year', type=int, required=True, help='Year (e.g., 2025)')
    parser.add_argument('--circuit', type=str, required=True, help='Circuit name (e.g., Monaco)')
    args = parser.parse_args()
    year = args.year
    circuit = args.circuit

    # Load data for the given track/year
    qual = load_qualifying_data(year, circuit)
    synthetic = False
    if qual.empty and year == 2025:
        print(f"No qualifying data found for {circuit} {year}. Generating synthetic entry from race lineup...")
        qual = generate_synthetic_qualifying(year, circuit)
        if qual.empty:
            print(f"No race lineup found for {circuit} {year}.")
            return
        synthetic = True

    features = ['year','circuit','driver_name','team_name','best_quali_time','qual_form_last3']
    cat_features = ['circuit','driver_name','team_name']
    num_features = [f for f in features if f not in cat_features]

    # Impute missing values
    for col in num_features:
        qual[col] = qual[col].fillna(0)
    for col in cat_features:
        qual[col] = qual[col].fillna('Unknown')

    # Load encoders and scaler
    encoders = joblib.load('model/encoders_qualifying.pkl')
    scaler = joblib.load('model/scaler_qualifying.pkl')
    for col in cat_features:
        le = encoders[col]
        qual[col] = qual[col].astype(str).apply(lambda x: x if x in le.classes_ else le.classes_[0])
        qual[col] = le.transform(qual[col])
    qual[num_features] = scaler.transform(qual[num_features])

    X = qual[features]

    # Try to load regression model first
    reg_model_path = 'model/xgb_qualifying_regression.model'
    clf_model_path = 'model/xgb_qualifying_top5.model'
    if os.path.exists(reg_model_path):
        model = joblib.load(reg_model_path)
        y_pred = model.predict(X)
        qual['predicted_position'] = y_pred
        qual = qual.sort_values('predicted_position')
        print(f"\n--- Predicted Qualifying Order for {circuit} {year} (Regression) ---")
        print("Pos | Driver                | Team")
        print("----+-----------------------+------------------------------")
        for i, row in enumerate(qual.itertuples(), 1):
            if synthetic:
                print(f"{i:2d}  | {row.driver_name_orig:<21} | {row.team_name_orig}")
            else:
                print(f"{i:2d}  | {row.driver_name:<21} | {row.team_name}")
    elif os.path.exists(clf_model_path):
        model = joblib.load(clf_model_path)
        y_prob = model.predict_proba(X)[:,1]
        qual['top5_prob'] = y_prob
        qual = qual.sort_values('top5_prob', ascending=False)
        print(f"\n--- Predicted Top 5 Qualifiers for {circuit} {year} (Classification) ---")
        print("Pos | Driver                | Team        | Top5 Prob")
        print("----+-----------------------+-------------+----------")
        for i, row in enumerate(qual.head(5).itertuples(), 1):
            if synthetic:
                print(f"{i:2d}  | {row.driver_name_orig:<21} | {row.team_name_orig:<11} | {row.top5_prob:.3f}")
            else:
                print(f"{i:2d}  | {row.driver_name:<21} | {row.team_name:<11} | {row.top5_prob:.3f}")
    else:
        print("No qualifying model found.")

if __name__ == "__main__":
    main() 