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

# --- XGBoost ---
print("\n--- Training XGBoost ---")
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)
print("XGBoost Test accuracy:", accuracy_score(y_test, xgb_pred))
print(classification_report(y_test, xgb_pred, target_names=['Not Top 5', 'Top 5']))
xgb_model.save_model('model/xgb_top5.model')

# --- LightGBM ---
print("\n--- Training LightGBM ---")
lgbm_model = lgb.LGBMClassifier()
lgbm_model.fit(X_train, y_train)
lgbm_pred = lgbm_model.predict(X_test)
print("LightGBM Test accuracy:", accuracy_score(y_test, lgbm_pred))
print(classification_report(y_test, lgbm_pred, target_names=['Not Top 5', 'Top 5']))
lgbm_model.booster_.save_model('model/lgbm_top5.txt')

# --- CatBoost ---
print("\n--- Training CatBoost ---")
cat_model = cb.CatBoostClassifier(verbose=0, cat_features=cat_features)
cat_model.fit(X_train, y_train)
cat_pred = cat_model.predict(X_test)
print("CatBoost Test accuracy:", accuracy_score(y_test, cat_pred))
print(classification_report(y_test, cat_pred, target_names=['Not Top 5', 'Top 5']))
cat_model.save_model('model/catboost_top5.cbm')

# --- Keras Neural Network (Advanced Ensemble Architecture) ---
print("\n--- Training Keras NN (Advanced Ensemble Architecture) ---")
advanced_models = AdvancedF1Models()
advanced_training = AdvancedTraining()

# Build the model using the advanced architecture
nn_model = advanced_models.build_ensemble_model(input_shape=(X_train.shape[1],))

# Get custom callbacks for smarter training
callbacks = advanced_training.create_custom_callbacks()

# Compile the model
nn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model with a higher epoch count, relying on EarlyStopping
print("Training advanced NN model (will stop early if no improvement)...")
history = nn_model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=0, callbacks=callbacks)

print(f"Advanced NN training complete. Stopped at epoch: {len(history.history['loss'])}")

# Evaluate the advanced model
y_pred_nn = (nn_model.predict(X_test) > 0.5).astype(int)
print("Keras Advanced NN Test accuracy:", accuracy_score(y_test, y_pred_nn))
print(classification_report(y_test, y_pred_nn, target_names=['Not Top 5', 'Top 5']))

# Save the newly trained advanced model
nn_model.save('model/pre_race_model_top5.keras')

# --- 4. Stacking Ensemble ---
print("\n--- Training Stacking Meta-Model ---")
# Get predictions from base models on the test set
xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
lgbm_probs = lgbm_model.predict_proba(X_test)[:, 1]
cat_probs = cat_model.predict_proba(X_test)[:, 1]
nn_probs = nn_model.predict(X_test).flatten()

# Create a new dataset for the meta-model
stack_X_test = np.vstack([xgb_probs, lgbm_probs, cat_probs, nn_probs]).T

# Also need predictions on the training set to train the meta-model
xgb_probs_train = xgb_model.predict_proba(X_train)[:, 1]
lgbm_probs_train = lgbm_model.predict_proba(X_train)[:, 1]
cat_probs_train = cat_model.predict_proba(X_train)[:, 1]
nn_probs_train = nn_model.predict(X_train).flatten()
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