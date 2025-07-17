import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import xgboost as xgb
from sklearn.model_selection import GridSearchCV

# Load data
DATA_PATH = 'data/pre_race_features.csv'
df = pd.read_csv(DATA_PATH)

# Drop rows with missing target or grid position
features = [
    'grid_position', 'qualifying_lap_time', 'air_temperature', 'humidity', 'rainfall',
    'track_temperature', 'wind_speed', 'team_name', 'driver_name', 'circuit', 'country_code',
    'driver_form_last3', 'team_form_last3', 'qualifying_gap_to_pole', 'teammate_grid_delta',
    'track_type', 'overtaking_difficulty',
    'driver_championship_position', 'team_championship_position', 'driver_points_season', 'team_points_season'
]
target = 'finishing_position'

# Fill missing championship/points features with -1 before dropping rows
for col in ['driver_championship_position', 'team_championship_position', 'driver_points_season', 'team_points_season']:
    if col in df.columns:
        df[col] = df[col].fillna(-1)

# Now dropna for the rest
print("Missing values per feature BEFORE dropna:")
print(df[features + [target]].isnull().sum())
df = df.dropna(subset=features + [target])
if df.empty:
    print("No data left after dropping rows with missing values!")
    print("Number of missing values per feature:")
    print(df[features + [target]].isnull().sum())
    exit(1)

# Filter out rows where finishing_position is not a valid integer (e.g., DQ, DNS, DNF)
def is_int_str(x):
    try:
        int(x)
        return True
    except:
        return False
invalid_rows = ~df[target].apply(is_int_str)
if invalid_rows.any():
    print(f"Dropping {invalid_rows.sum()} rows with non-integer finishing_position")
    df = df[~invalid_rows]

df['finishing_position'] = df['finishing_position'].astype(int)

# Add is_top5 column
TOP_N = 5
df['is_top5'] = (df['finishing_position'] <= TOP_N).astype(int)

# Encode categoricals
cat_features = ['team_name', 'driver_name', 'circuit', 'country_code', 'track_type']
encoders = {}
for col in cat_features:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

# Scale numerics
num_features = [f for f in features if f not in cat_features]
scaler = StandardScaler()
df[num_features] = scaler.fit_transform(df[num_features])

# Train/test split
X = df[features]
y = df['is_top5']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- XGBoost ---
import xgboost as xgb
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)
print("\nXGBoost Test accuracy:", accuracy_score(y_test, xgb_pred))
print("XGBoost Confusion matrix:\n", confusion_matrix(y_test, xgb_pred))
print("XGBoost Classification report:\n", classification_report(y_test, xgb_pred, target_names=['Not Top 5', 'Top 5']))
xgb_model.save_model('model/xgb_top5.model')

# --- LightGBM ---
import lightgbm as lgb
lgbm_model = lgb.LGBMClassifier()
lgbm_model.fit(X_train, y_train)
lgbm_pred = lgbm_model.predict(X_test)
print("\nLightGBM Test accuracy:", accuracy_score(y_test, lgbm_pred))
print("LightGBM Confusion matrix:\n", confusion_matrix(y_test, lgbm_pred))
print("LightGBM Classification report:\n", classification_report(y_test, lgbm_pred, target_names=['Not Top 5', 'Top 5']))
lgbm_model.booster_.save_model('model/lgbm_top5.txt')

# Build model
model = keras.Sequential([
    layers.Input(shape=(X.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train
model.fit(X_train, y_train, epochs=40, batch_size=32, validation_split=0.2, verbose=2)

# Evaluate
y_pred = (model.predict(X_test) > 0.5).astype(int)
print("\nTest accuracy:", accuracy_score(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=['Not Top 5', 'Top 5']))

# Sample predictions
sample = X_test.iloc[:10]
sample_pred = model.predict(sample)
print("\nSample predictions:")
for idx, prob in zip(sample.index, sample_pred):
    driver_name = df.loc[idx, 'driver_name'] if idx in df.index else 'Unknown'
    actual = y_test.loc[idx] if idx in y_test.index else 'Unknown'
    print(f"Driver: {driver_name}, Prob Top 5: {prob[0]:.2%}, Actual: {actual}")

# Save model
os.makedirs('model', exist_ok=True)
model.save('model/pre_race_model_top5.keras')

# Save encoders/scaler for prediction script
import joblib
joblib.dump(encoders, 'model/encoders_top5.pkl')
joblib.dump(scaler, 'model/scaler_top5.pkl')

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1],
    'subsample': [0.8, 1.0]
}

base_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', verbosity=0)
grid_search = GridSearchCV(base_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

xgb_model = grid_search.best_estimator_
print("Best hyperparameters:", grid_search.best_params_)