import pandas as pd
import joblib
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# Load data and features
DATA_PATH = 'data/pre_race_features.csv'
df = pd.read_csv(DATA_PATH)
df = df[(df['year'] >= 2010) & (df['year'] <= 2025)]

features = [
    'grid_position', 'qualifying_lap_time', 'air_temperature', 'humidity', 'rainfall',
    'track_temperature', 'wind_speed', 'team_name', 'driver_name', 'circuit', 'country_code',
    'driver_form_last3', 'team_form_last3', 'qualifying_gap_to_pole', 'teammate_grid_delta',
    'track_type', 'overtaking_difficulty',
    'driver_championship_position', 'team_championship_position', 'driver_points_season', 'team_points_season'
]
target = 'finishing_position'

# Impute as in training
for col in ['driver_championship_position', 'team_championship_position', 'driver_points_season', 'team_points_season']:
    if col in df.columns:
        df[col] = df[col].fillna(-1)
cat_features = ['team_name', 'driver_name', 'circuit', 'country_code', 'track_type']
num_features = [f for f in features if f not in cat_features]
for col in num_features:
    if col in df.columns:
        median = df[col].median()
        df[col] = df[col].fillna(median)
for col in cat_features:
    if col in df.columns:
        mode = df[col].mode()[0]
        df[col] = df[col].fillna(mode)

def is_int_str(x):
    try:
        int(x)
        return True
    except:
        return False
invalid_rows = ~df[target].apply(is_int_str)
if invalid_rows.any():
    df = df[~invalid_rows]
df['finishing_position'] = df['finishing_position'].astype(int)
TOP_N = 5
df['is_top5'] = (df['finishing_position'] <= TOP_N).astype(int)

# Encode categoricals
encoders = joblib.load('model/encoders_top5.pkl')
for col in cat_features:
    le = encoders[col]
    df[col] = le.transform(df[col].astype(str))

# Scale numerics
scaler = joblib.load('model/scaler_top5.pkl')
df[num_features] = scaler.transform(df[num_features])

# Train/test split (same as before)
X = df[features]
y = df['is_top5']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Load models
xgb_model = xgb.XGBClassifier()
xgb_model.load_model('model/xgb_top5.model')
lgbm_model = lgb.Booster(model_file='model/lgbm_top5.txt')
cat_model = cb.CatBoostClassifier()
cat_model.load_model('model/catboost_top5.cbm')

# Predict probabilities
xgb_probs = xgb_model.predict_proba(X_test)[:,1]
lgbm_probs = lgbm_model.predict(X_test)
cat_probs = cat_model.predict_proba(X_test)[:,1]

# Average ensemble
avg_probs = (xgb_probs + lgbm_probs + cat_probs) / 3
ensemble_pred = (avg_probs > 0.5).astype(int)

# Evaluate
print("\nEnsemble (Averaging) Test accuracy:", accuracy_score(y_test, ensemble_pred))
print("Ensemble Confusion matrix:\n", confusion_matrix(y_test, ensemble_pred))
print("Ensemble Classification report:\n", classification_report(y_test, ensemble_pred, target_names=['Not Top 5', 'Top 5'])) 