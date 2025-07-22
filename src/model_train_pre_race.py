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

# Prepare data for modeling
X = df[features]
y = (df[target] <= 5).astype(int)  # Top 5 classification

# Final robust NaN imputation for all features
print(f"NaNs in X before final imputation: {X.isna().sum().sum()}")
for col in X.select_dtypes(include=[np.number]).columns:
    X[col] = X[col].fillna(X[col].median())
for col in X.select_dtypes(include=['object']).columns:
    mode = X[col].mode()[0] if not X[col].mode().empty else 'Unknown'
    X[col] = X[col].fillna(mode)
print("NaNs per column after median/mode imputation:")
print(X.isna().sum())
# Fill any remaining NaNs with 0 (numeric) or 'Unknown' (categorical)
for col in X.columns:
    if X[col].isna().any():
        if X[col].dtype.kind in 'biufc':
            X[col] = X[col].fillna(0)
        else:
            X[col] = X[col].fillna('Unknown')
print("NaNs per column after final fill:")
print(X.isna().sum())
# Optionally, drop columns that are all NaN (should be none after above)
all_nan_cols = X.columns[X.isna().all()]
if len(all_nan_cols) > 0:
    print(f"Dropping columns that are all NaN: {list(all_nan_cols)}")
    X = X.drop(columns=all_nan_cols)


# Ensure target has both classes
if y.nunique() < 2:
    print('ERROR: Only one class present in target variable. Check data filtering.')
    exit(1)

# Print class distribution in each fold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for i, (train_idx, test_idx) in enumerate(cv.split(X, y)):
    print(f"Fold {i+1} class distribution:")
    print(pd.Series(y.iloc[train_idx]).value_counts(normalize=True))

# --- XGBoost with GridSearchCV ---
xgb_params = {'max_depth': [3, 5, 7], 'learning_rate': [0.01, 0.1], 'n_estimators': [100, 200]}
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_grid = GridSearchCV(xgb_model, xgb_params, cv=cv, scoring='accuracy', n_jobs=-1)
xgb_grid.fit(X, y)
print("Best XGBoost params:", xgb_grid.best_params_)
xgb_pred = cross_val_predict(xgb_grid.best_estimator_, X, y, cv=cv, method='predict')
print("XGBoost CV accuracy:", accuracy_score(y, xgb_pred))
print(classification_report(y, xgb_pred, target_names=['Not Top 5', 'Top 5']))
xgb_grid.best_estimator_.save_model('model/xgb_top5.model')

# --- LightGBM with GridSearchCV ---
lgbm_params = {'max_depth': [3, 5, 7], 'learning_rate': [0.01, 0.1], 'n_estimators': [100, 200]}
lgbm_model = lgb.LGBMClassifier()
lgbm_grid = GridSearchCV(lgbm_model, lgbm_params, cv=cv, scoring='accuracy', n_jobs=-1)
lgbm_grid.fit(X, y)
print("Best LightGBM params:", lgbm_grid.best_params_)
lgbm_pred = cross_val_predict(lgbm_grid.best_estimator_, X, y, cv=cv, method='predict')
print("LightGBM CV accuracy:", accuracy_score(y, lgbm_pred))
print(classification_report(y, lgbm_pred, target_names=['Not Top 5', 'Top 5']))
lgbm_grid.best_estimator_.booster_.save_model('model/lgbm_top5.txt')

# --- CatBoost with GridSearchCV ---
cat_params = {'depth': [3, 5, 7], 'learning_rate': [0.01, 0.1], 'iterations': [100, 200]}
cat_model = cb.CatBoostClassifier(verbose=0, cat_features=cat_features)
cat_grid = GridSearchCV(cat_model, cat_params, cv=cv, scoring='accuracy', n_jobs=-1)
cat_grid.fit(X, y)
print("Best CatBoost params:", cat_grid.best_params_)
cat_pred = cross_val_predict(cat_grid.best_estimator_, X, y, cv=cv, method='predict')
print("CatBoost CV accuracy:", accuracy_score(y, cat_pred))
print(classification_report(y, cat_pred, target_names=['Not Top 5', 'Top 5']))
cat_grid.best_estimator_.save_model('model/catboost_top5.cbm')

# --- Neural Net with KerasClassifier and GridSearchCV ---
def build_nn_model(optimizer='adam', dropout=0.2):
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(X.shape[1],)),
        layers.BatchNormalization(),
        layers.Dropout(dropout),
        layers.Dense(32, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(dropout),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model
from scikeras.wrappers import KerasClassifier
nn_model = KerasClassifier(model=build_nn_model, epochs=30, batch_size=32, verbose=0)
# Use model__ prefix for scikeras param grid
nn_params = {'model__optimizer': ['adam', 'rmsprop'], 'model__dropout': [0.2, 0.4]}
nn_grid = GridSearchCV(nn_model, nn_params, cv=3, scoring='accuracy', n_jobs=-1)
nn_grid.fit(X, y)
print("Best NN params:", nn_grid.best_params_)
nn_pred = cross_val_predict(nn_grid.best_estimator_, X, y, cv=cv, method='predict')
print("Neural Net CV accuracy:", accuracy_score(y, nn_pred))
print(classification_report(y, nn_pred, target_names=['Not Top 5', 'Top 5']))
joblib.dump(nn_grid.best_estimator_, 'model/pre_race_model_top5.keras')

# --- Ensemble Stacking ---
print("\n--- Training Stacking Meta-Model ---")
# Get cross-validated probabilities for stacking
xgb_probs = cross_val_predict(xgb_grid.best_estimator_, X, y, cv=cv, method='predict_proba')[:, 1]
lgbm_probs = cross_val_predict(lgbm_grid.best_estimator_, X, y, cv=cv, method='predict_proba')[:, 1]
cat_probs = cross_val_predict(cat_grid.best_estimator_, X, y, cv=cv, method='predict_proba')[:, 1]
nn_probs = cross_val_predict(nn_grid.best_estimator_, X, y, cv=cv, method='predict_proba')[:, 1]
stack_X = np.vstack([xgb_probs, lgbm_probs, cat_probs, nn_probs]).T

# Train meta-model
meta_model = LogisticRegression()
meta_model.fit(stack_X, y)
meta_pred = meta_model.predict(stack_X)
print("Stacking Meta-Model CV accuracy:", accuracy_score(y, meta_pred))
print(classification_report(y, meta_pred, target_names=['Not Top 5', 'Top 5']))
joblib.dump(meta_model, 'model/meta_model_logreg.pkl')

print("\nAll models trained and saved successfully.")

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
    y = (df['positionOrder'] <= 5).astype(int)  # Example: predict top 5 finish
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