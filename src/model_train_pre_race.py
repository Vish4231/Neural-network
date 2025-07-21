import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score, GridSearchCV
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
df = df[(df['year'] >= 2015) & (df['year'] <= 2025)].copy()

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

# Ensure target has both classes
if y.nunique() < 2:
    print('ERROR: Only one class present in target variable. Check data filtering.')
    exit(1)

# Train/test split using stratification to maintain class balance
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")

# --- XGBoost with GridSearchCV ---
xgb_params = {'max_depth': [3, 5, 7], 'learning_rate': [0.01, 0.1], 'n_estimators': [100, 200]}
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_grid = GridSearchCV(xgb_model, xgb_params, cv=5, scoring='accuracy', n_jobs=-1)
xgb_grid.fit(X_train, y_train)
print("Best XGBoost params:", xgb_grid.best_params_)
xgb_pred = xgb_grid.predict(X_test)
print("XGBoost Test accuracy:", accuracy_score(y_test, xgb_pred))
print(classification_report(y_test, xgb_pred, target_names=['Not Top 5', 'Top 5']))
xgb_grid.best_estimator_.save_model('model/xgb_top5.model')

# --- LightGBM with GridSearchCV ---
lgbm_params = {'max_depth': [3, 5, 7], 'learning_rate': [0.01, 0.1], 'n_estimators': [100, 200]}
lgbm_model = lgb.LGBMClassifier()
lgbm_grid = GridSearchCV(lgbm_model, lgbm_params, cv=5, scoring='accuracy', n_jobs=-1)
lgbm_grid.fit(X_train, y_train)
print("Best LightGBM params:", lgbm_grid.best_params_)
lgbm_pred = lgbm_grid.predict(X_test)
print("LightGBM Test accuracy:", accuracy_score(y_test, lgbm_pred))
print(classification_report(y_test, lgbm_pred, target_names=['Not Top 5', 'Top 5']))
lgbm_grid.best_estimator_.booster_.save_model('model/lgbm_top5.txt')

# --- CatBoost with GridSearchCV ---
cat_params = {'depth': [3, 5, 7], 'learning_rate': [0.01, 0.1], 'iterations': [100, 200]}
cat_model = cb.CatBoostClassifier(verbose=0, cat_features=cat_features)
cat_grid = GridSearchCV(cat_model, cat_params, cv=5, scoring='accuracy', n_jobs=-1)
cat_grid.fit(X_train, y_train)
print("Best CatBoost params:", cat_grid.best_params_)
cat_pred = cat_grid.predict(X_test)
print("CatBoost Test accuracy:", accuracy_score(y_test, cat_pred))
print(classification_report(y_test, cat_pred, target_names=['Not Top 5', 'Top 5']))
cat_grid.best_estimator_.save_model('model/catboost_top5.cbm')

# --- Neural Net with KerasClassifier and GridSearchCV ---
def build_nn_model(optimizer='adam', dropout=0.2):
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
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
nn_params = {'optimizer': ['adam', 'rmsprop'], 'dropout': [0.2, 0.4]}
nn_grid = GridSearchCV(nn_model, nn_params, cv=3, scoring='accuracy', n_jobs=-1)
nn_grid.fit(X_train, y_train)
print("Best NN params:", nn_grid.best_params_)
nn_pred = nn_grid.predict(X_test)
print("Neural Net Test accuracy:", accuracy_score(y_test, nn_pred))
print(classification_report(y_test, nn_pred, target_names=['Not Top 5', 'Top 5']))
joblib.dump(nn_grid.best_estimator_, 'model/pre_race_model_top5.keras')

# --- 4. Stacking Ensemble ---
print("\n--- Training Stacking Meta-Model ---")
# Get predictions from base models on the test set
xgb_probs = xgb_grid.predict_proba(X_test)[:, 1]
lgbm_probs = lgbm_grid.predict_proba(X_test)[:, 1]
cat_probs = cat_grid.predict_proba(X_test)[:, 1]
nn_probs = nn_grid.predict(X_test).flatten()

# Create a new dataset for the meta-model
stack_X_test = np.vstack([xgb_probs, lgbm_probs, cat_probs, nn_probs]).T

# Also need predictions on the training set to train the meta-model
xgb_probs_train = xgb_grid.predict_proba(X_train)[:, 1]
lgbm_probs_train = lgbm_grid.predict_proba(X_train)[:, 1]
cat_probs_train = cat_grid.predict_proba(X_train)[:, 1]
nn_probs_train = nn_grid.predict(X_train).flatten()
stack_X_train = np.vstack([xgb_probs_train, lgbm_probs_train, cat_probs_train, nn_probs_train]).T

# Train the meta-model
meta_model = LogisticRegression()

# Impute any NaNs in stack_X_train with column means
if np.isnan(stack_X_train).any():
    print('Warning: NaNs found in stack_X_train, imputing with column means.')
    # If any column is all NaN, fill with zeros first
    for i in range(stack_X_train.shape[1]):
        if np.isnan(stack_X_train[:, i]).all():
            stack_X_train[:, i] = 0.0
    col_means = np.nanmean(stack_X_train, axis=0)
    inds = np.where(np.isnan(stack_X_train))
    stack_X_train[inds] = np.take(col_means, inds[1])

# Impute any NaNs in stack_X_test with column means
if np.isnan(stack_X_test).any():
    print('Warning: NaNs found in stack_X_test, imputing with column means.')
    for i in range(stack_X_test.shape[1]):
        if np.isnan(stack_X_test[:, i]).all():
            stack_X_test[:, i] = 0.0
    col_means_test = np.nanmean(stack_X_test, axis=0)
    inds_test = np.where(np.isnan(stack_X_test))
    stack_X_test[inds_test] = np.take(col_means_test, inds_test[1])

meta_model.fit(stack_X_train, y_train)
meta_pred = meta_model.predict(stack_X_test)
print("Stacking Meta-Model Test accuracy:", accuracy_score(y_test, meta_pred))
print(classification_report(y_test, meta_pred, target_names=['Not Top 5', 'Top 5']))

# Save the meta-model
joblib.dump(meta_model, 'model/meta_model_logreg.pkl')

print("\nAll models trained and saved successfully.")