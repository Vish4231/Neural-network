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
import catboost as cb
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from scikeras.wrappers import KerasClassifier

# ADVANCED DATA LOADING AND FEATURE ENGINEERING FROM archive/ (1950-2024)
import glob
# Load all relevant CSVs from archive/
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

# Feature: grid position, finishing position, Q1/Q2/Q3 (as seconds), team, driver, year, circuit
# Advanced: rolling form (last 3/5), pit stop count, avg lap time, feature interactions, etc.
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
def add_rolling_form(df, form_window=3, form_col='driver_form_last3', group_col='driverId'):
    df = df.sort_values(['year', 'raceId'])
    df[form_col] = (
        df.groupby(group_col)['positionOrder']
        .apply(lambda x: x.shift(1).rolling(form_window, min_periods=1).mean())
        .reset_index(level=0, drop=True)
    )
    return df
for window in [3,5]:
    results = add_rolling_form(results, form_window=window, form_col=f'driver_form_last{window}', group_col='driverId')
    results = add_rolling_form(results, form_window=window, form_col=f'team_form_last{window}', group_col='constructorId')

# Feature interactions: grid - qual pos, pit_stop_count * avg_lap_time, etc.
results['grid_vs_qual'] = results['grid'] - results['position_qual']
results['pit_lap_interaction'] = results['pit_stop_count'] * results['avg_lap_time']

# Select and rename features to match model pipeline
results['driver_name'] = results['driver_forename'] + ' ' + results['driver_surname']
features = [
    'grid', 'q1', 'q2', 'q3', 'pit_stop_count', 'avg_lap_time',
    'team_name', 'driver_name', 'circuit', 'year',
    'driver_form_last3', 'driver_form_last5', 'team_form_last3', 'team_form_last5',
    'grid_vs_qual', 'pit_lap_interaction'
]
target = 'positionOrder'

cat_features = ['team_name', 'driver_name', 'circuit']
num_features = [f for f in features if f not in cat_features]

# Filter for years 1950-2024
results = results[(results['year'] >= 1950) & (results['year'] <= 2024)]

# Final DataFrame for training
# (rest of pipeline: encoding, scaling, imputation, etc. as before)
df = results.copy()

# Diagnostic: print distribution of positionOrder before any filtering
print('Distribution of positionOrder before filtering:')
print(df['positionOrder'].value_counts().sort_index())

# Only drop rows where positionOrder is missing or not an integer
invalid_rows = df['positionOrder'].isnull() | (~df['positionOrder'].apply(lambda x: isinstance(x, (int, float)) or (isinstance(x, str) and x.isdigit())))
if invalid_rows.any():
    print(f'Dropping {invalid_rows.sum()} rows with missing or non-integer positionOrder')
    df = df[~invalid_rows]

df['positionOrder'] = df['positionOrder'].astype(int)

# Diagnostic: print class balance for top-5 vs not top-5
print('Class balance for top-5 (<=5) vs not top-5:')
print((df['positionOrder'] <= 5).value_counts())

# Only proceed if both classes are present
if (df['positionOrder'] <= 5).nunique() < 2:
    print('ERROR: Only one class present after filtering. Check data and pipeline.')
    exit(1)

# Ensure Q1, Q2, Q3 columns exist for all rows (fill with NaN if missing)
for col in ['Q1', 'Q2', 'Q3']:
    if col not in df.columns:
        df[col] = np.nan

# Fill missing championship/points features with -1 before imputation
for col in ['driver_championship_position', 'team_championship_position', 'driver_points_season', 'team_points_season']:
    if col in df.columns:
        df[col] = df[col].fillna(-1)

# Print missing values before imputation
print("Missing values per feature BEFORE imputation:")
print(df[features + [target]].isnull().sum())
print(f"Rows before imputation: {len(df)}")

# Impute missing values
# Use only the new features and target
features = [
    'grid', 'q1', 'q2', 'q3', 'pit_stop_count', 'avg_lap_time',
    'team_name', 'driver_name', 'circuit', 'year',
    'driver_form_last3', 'driver_form_last5', 'team_form_last3', 'team_form_last5',
    'grid_vs_qual', 'pit_lap_interaction'
]
target = 'positionOrder'

# Impute all numeric features with median, categorical with mode, and fill NaN for form/Q1/Q2/Q3 with 0 (or -1 for championship/points)
for col in num_features:
    if col in df.columns:
        median = df[col].median()
        df[col] = df[col].fillna(median)
for col in cat_features:
    if col in df.columns:
        mode = df[col].mode()[0]
        df[col] = df[col].fillna(mode)
# For rolling form features and Q1/Q2/Q3, fill NaN with 0 (no history = neutral)
for col in ['Q1', 'Q2', 'Q3', 'driver_form_last3', 'driver_form_last5', 'team_form_last3', 'team_form_last5']:
    if col in df.columns:
        df[col] = df[col].fillna(0)
# For championship/points features, fill NaN with -1
for col in ['driver_championship_position', 'team_championship_position', 'driver_points_season', 'team_points_season']:
    if col in df.columns:
        df[col] = df[col].fillna(-1)

# --- IMPUTE WEATHER/CONTEXTUAL FEATURES ---
def impute_weather(df):
    defaults = {
        'air_temperature': 22,
        'humidity': 60,
        'rainfall': 0,
        'track_temperature': 30,
        'wind_speed': 5
    }
    for col, default in defaults.items():
        if col in df.columns:
            df[col] = df[col].fillna(default)
    return df

df = impute_weather(df)

# Print missing values after imputation
print("Missing values per feature AFTER imputation:")
print(df[features + [target]].isnull().sum())
print(f"Rows after imputation: {len(df)}")

# After imputation, check class balance
print('Class balance after imputation:')
print((df['positionOrder'] <= 5).value_counts())
if (df['positionOrder'] <= 5).nunique() < 2:
    print('ERROR: Only one class present after imputation. Check imputation logic.')
    exit(1)

# Remove dropna (was here):
# df = df.dropna(subset=features + [target])
if df.empty:
    print("No data left after imputation!")
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

# Remove any reference to 'finishing_position' (use 'positionOrder' everywhere)
# All logic for top 5, target, and reporting already uses 'positionOrder'
# No need to cast 'finishing_position' or add 'is_top5' column separately

# Add rolling form features for drivers and teams (last 3 and 5 races)
def add_rolling_form(df, form_window=3, form_col='driver_form_last3', group_col='driver_name'):
    df = df.sort_values(['year', 'circuit'])
    df[form_col] = (
        df.groupby(group_col)['positionOrder']
        .apply(lambda x: x.shift(1).rolling(form_window, min_periods=1).mean())
        .reset_index(level=0, drop=True)
    )
    return df

def add_rolling_team_form(df, form_window=3, form_col='team_form_last3', group_col='team_name'):
    df = df.sort_values(['year', 'circuit'])
    df[form_col] = (
        df.groupby(group_col)['positionOrder']
        .apply(lambda x: x.shift(1).rolling(form_window, min_periods=1).mean())
        .reset_index(level=0, drop=True)
    )
    return df

# Add last 3 and last 5 form features for drivers and teams
for window in [3, 5]:
    df = add_rolling_form(df, form_window=window, form_col=f'driver_form_last{window}', group_col='driver_name')
    df = add_rolling_team_form(df, form_window=window, form_col=f'team_form_last{window}', group_col='team_name')

# Encode categoricals
cat_features = ['team_name', 'driver_name', 'circuit']
encoders = {}
for col in cat_features:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

# After encoding, check class balance
print('Class balance after encoding:')
print((df['positionOrder'] <= 5).value_counts())
if (df['positionOrder'] <= 5).nunique() < 2:
    print('ERROR: Only one class present after encoding. Check encoding logic.')
    exit(1)

# Scale numerics
num_features = [f for f in features if f not in cat_features]
scaler = StandardScaler()
df.loc[:, num_features] = scaler.fit_transform(df[num_features])

# After scaling, check class balance
print('Class balance after scaling:')
print((df['positionOrder'] <= 5).value_counts())
if (df['positionOrder'] <= 5).nunique() < 2:
    print('ERROR: Only one class present after scaling. Check scaling logic.')
    exit(1)

# Save encoders and scaler
joblib.dump(encoders, 'model/encoders_top5.pkl')
joblib.dump(scaler, 'model/scaler_top5.pkl')

# Final NaN check: ensure all features are numeric and fill any remaining NaNs with 0
for col in features:
    if col not in df.columns:
        df[col] = 0
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

# Train/test split
X = df[features]
y = (df['positionOrder'] <= 5).astype(int)  # Top 5 classification

# Check class distribution before splitting
print('Target class distribution (Top 5 = 1, Not Top 5 = 0):')
print(y.value_counts())
if y.nunique() < 2:
    print('ERROR: Only one class present in target variable. Check data filtering and ensure both top-5 and non-top-5 finishers are included.')
    exit(1)

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

# --- CatBoost ---
cat_model = cb.CatBoostClassifier(verbose=0)
cat_model.fit(X_train, y_train)
cat_pred = cat_model.predict(X_test)
cat_probs = cat_model.predict_proba(X_test)[:,1]
print("\nCatBoost Test accuracy:", accuracy_score(y_test, cat_pred))
print("CatBoost Confusion matrix:\n", confusion_matrix(y_test, cat_pred))
print("CatBoost Classification report:\n", classification_report(y_test, cat_pred, target_names=['Not Top 5', 'Top 5']))
cat_model.save_model('model/catboost_top5.cbm')

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

# For stacking/blending, save all model outputs
stacking_df = pd.DataFrame({
    'xgb': xgb_model.predict_proba(X_test)[:,1],
    'lgbm': lgbm_model.predict_proba(X_test)[:,1],
    'cat': cat_probs,
    'nn': model.predict(X_test).flatten(),
    'target': y_test.values
})
stacking_df.to_csv('model/stacking_outputs.csv', index=False)
print("\nSaved stacking outputs for meta-model training.")

# --- TIME-SERIES CROSS-VALIDATION ---
print("\nTime-Series Cross-Validation (XGBoost, LightGBM, CatBoost, NN, Meta-Model):")
ts_cv = TimeSeriesSplit(n_splits=5)

# XGBoost CV
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_cv_scores = cross_val_score(xgb_model, X, y, cv=ts_cv, scoring='accuracy')
print(f"XGBoost CV accuracy: {xgb_cv_scores.mean():.3f} ± {xgb_cv_scores.std():.3f}")

# LightGBM CV
lgbm_model = lgb.LGBMClassifier()
lgbm_cv_scores = cross_val_score(lgbm_model, X, y, cv=ts_cv, scoring='accuracy')
print(f"LightGBM CV accuracy: {lgbm_cv_scores.mean():.3f} ± {lgbm_cv_scores.std():.3f}")

# CatBoost CV
cat_model = cb.CatBoostClassifier(verbose=0)
cat_cv_scores = cross_val_score(cat_model, X, y, cv=ts_cv, scoring='accuracy')
print(f"CatBoost CV accuracy: {cat_cv_scores.mean():.3f} ± {cat_cv_scores.std():.3f}")

# Neural Net CV (use KerasClassifier wrapper)
def make_nn():
    model = keras.Sequential([
        layers.Input(shape=(X.shape[1],)),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

nn_model = KerasClassifier(build_fn=make_nn, epochs=20, batch_size=32, verbose=0)
try:
    nn_cv_scores = cross_val_score(nn_model, X, y, cv=ts_cv, scoring='accuracy')
    print(f"Neural Net CV accuracy: {nn_cv_scores.mean():.3f} ± {nn_cv_scores.std():.3f}")
except Exception as e:
    print(f"Neural Net CV failed: {e}")

# --- ENSEMBLE META-MODEL (STACKING) ---
# Fit base models on full train set
xgb_model.fit(X, y)
lgbm_model.fit(X, y)
cat_model.fit(X, y)
nn_model.fit(X, y)
# Get base model outputs
xgb_probs = xgb_model.predict_proba(X)[:,1]
lgbm_probs = lgbm_model.predict_proba(X)[:,1]
cat_probs = cat_model.predict_proba(X)[:,1]
nn_probs = nn_model.predict(X)
# Stack as features
stack_X = np.vstack([xgb_probs, lgbm_probs, cat_probs, nn_probs]).T
meta_model = LogisticRegression()
meta_cv_scores = cross_val_score(meta_model, stack_X, y, cv=ts_cv, scoring='accuracy')
print(f"Meta-Model (LogReg) CV accuracy: {meta_cv_scores.mean():.3f} ± {meta_cv_scores.std():.3f}")
# Fit meta-model on all data
meta_model.fit(stack_X, y)
# Save meta-model
joblib.dump(meta_model, 'model/meta_model_logreg.pkl')