import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score, GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import joblib
from sklearn.linear_model import LogisticRegression
from scikeras.wrappers import KerasClassifier

# Import the centralized feature engineering function
from feature_engineering import load_and_engineer_features
from advanced_models import AdvancedF1Models, AdvancedTraining

# --- 1. Load and Prepare Data ---
print("Loading and engineering features...")
df = load_and_engineer_features()
print("Features loaded successfully.")

# Define features and target
target = 'positionOrder'
features = [col for col in df.columns if col not in [target, 'raceId']]
cat_features = ['team_name', 'driver_name', 'circuit', 'track_type']
num_features = [f for f in features if f not in cat_features and f != 'year']

# Filter for years 2015-2025
print("Filtering data for years 2015-2025...")
df = df[(df['year'] >= 2015) & (df['year'] <= 2025)].copy()

# Print unique values for all track features to confirm they are present and dynamic
track_feature_cols = [
    'length_km', 'turns', 'elevation', 'drs_zones', 'grip', 'rain_prob', 'track_type',
    'overtaking_difficulty', 'pit_lane_time_loss', 'avg_lap_speed', 'surface_type',
    'track_width', 'safety_car_prob', 'tyre_deg', 'corner_type_dist'
]
for col in track_feature_cols:
    if col in df.columns:
        print(f"Unique values for {col}: {df[col].unique()}")

# --- 2. Preprocessing ---
print("\nStarting preprocessing...")

# Impute missing values
# Numeric features with median
for col in num_features:
    if col in df.columns:
        median = df[col].median()
        df[col] = df[col].fillna(median)
# Categorical features with mode
for col in cat_features:
    if col in df.columns:
        mode_series = df[col].mode()
        if not mode_series.empty:
            mode = mode_series[0]
        else:
            mode = "Unknown"
        df[col] = df[col].fillna(mode)

print("Imputation complete.")

# Encode categoricals
# Encode categoricals
encoders = {}
for col in cat_features:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le
print("Categorical encoding complete.")

# Scale numerics
scaler = StandardScaler()
df.loc[:, num_features] = scaler.fit_transform(df[num_features])
print("Numeric scaling complete.")

# Save encoders and scaler
os.makedirs('model', exist_ok=True)
joblib.dump(encoders, 'model/encoders_top5.pkl')
joblib.dump(scaler, 'model/scaler_top5.pkl')
print("Encoders and scaler saved.")

# --- 3. Model Training ---
print("\nStarting model training...")

# Train separate models per track
unique_tracks = df['circuit'].unique()
for track in unique_tracks:
    print(f"\nTraining models for track: {track}")
    track_df = df[df['circuit'] == track]
    X_track = track_df[features]
    y_track = (track_df[target] <= 5).astype(int)

    # Impute missing values
    for col in X_track.select_dtypes(include=[np.number]).columns:
        X_track[col] = X_track[col].fillna(X_track[col].median())
    for col in X_track.select_dtypes(include=['object']).columns:
        mode = X_track[col].mode()[0] if not X_track[col].mode().empty else 'Unknown'
        X_track[col] = X_track[col].fillna(mode)

    # Ensure target has both classes
    if y_track.nunique() < 2:
        print(f"Skipping track {track} due to insufficient class variety.")
        continue

    # Print class distribution in each fold
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for i, (train_idx, test_idx) in enumerate(cv.split(X_track, y_track)):
        print(f"Fold {i+1} class distribution for {track}:")
        print(pd.Series(y_track.iloc[train_idx]).value_counts(normalize=True))

    # XGBoost with GridSearchCV
    xgb_params = {'max_depth': [3, 5, 7], 'learning_rate': [0.01, 0.1], 'n_estimators': [100, 200]}
    xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    xgb_grid = GridSearchCV(xgb_model, xgb_params, cv=cv, scoring='accuracy', n_jobs=-1)
    xgb_grid.fit(X_track, y_track)
    print(f"Best XGBoost params for {track}:", xgb_grid.best_params_)
    xgb_pred = cross_val_predict(xgb_grid.best_estimator_, X_track, y_track, cv=cv, method='predict')
    print(f"XGBoost CV accuracy for {track}:", accuracy_score(y_track, xgb_pred))
    print(classification_report(y_track, xgb_pred, target_names=['Not Top 5', 'Top 5']))
    original_track_name = encoders['circuit'].inverse_transform([track])[0]
    model_path = f'model/xgb_top5_{original_track_name.replace(" ", "_").lower()}.model'
    xgb_grid.best_estimator_.save_model(model_path)

    # Similarly train LightGBM, CatBoost, Neural Net, and stacking meta-model per track
    # For brevity, only XGBoost shown here; others can be added similarly

print("\nAll track-specific models trained and saved successfully.")

import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from feature_engineering import engineer_f1db_features, track_features

def train_race_model():
    df = pd.read_csv('data/f1db_merged_2010_2025.csv')
    df = engineer_f1db_features(df, track_features)
    # Select features and target
    features = ['driver_skill', 'driver_form_last3', 'team_form_last3', 'length_km', 'turns', 'elevation', 'drs_zones', 'grip', 'rain_prob']
    cat_features = ['driverId', 'constructorId']
    X = df[features + cat_features].fillna(0)
    y = (df['positionDisplayOrder'] <= 5).astype(int)  # Example: predict top 5 finish
    # Encode categorical features
    encoders = {}
    for col in cat_features:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le
    # Scale numerical features
    scaler = StandardScaler()
    X[features] = scaler.fit_transform(X[features])
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    # Save model and preprocessors
    joblib.dump(model, 'model/race_rf_model.pkl')
    joblib.dump(encoders, 'model/race_encoders.pkl')
    joblib.dump(scaler, 'model/race_scaler.pkl')
    print('Race model trained and saved.')

if __name__ == '__main__':
    train_race_model()