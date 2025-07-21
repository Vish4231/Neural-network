import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, accuracy_score, classification_report
import xgboost as xgb
import joblib

# Load qualifying data
qual_path = 'archive (1)/qualifying.csv'
qual = pd.read_csv(qual_path)

# Merge with race and driver info for more features if needed
races = pd.read_csv('archive (1)/races.csv')
drivers = pd.read_csv('archive (1)/drivers.csv')
constructors = pd.read_csv('archive (1)/constructors.csv')
qual = qual.merge(races[['raceId','year','name']], on='raceId', how='left')
qual = qual.merge(drivers[['driverId','surname','forename']], on='driverId', how='left')
qual = qual.merge(constructors[['constructorId','name']], on='constructorId', how='left', suffixes=('', '_team'))
qual = qual.rename(columns={'name':'circuit', 'name_team':'team_name', 'surname':'driver_surname', 'forename':'driver_forename'})

# Filter for years 2015-2025
qual = qual[(qual['year'] >= 2015) & (qual['year'] <= 2025)].copy()

# Feature engineering
qual['driver_name'] = qual['driver_forename'] + ' ' + qual['driver_surname']
for col in ['q1','q2','q3']:
    qual[col] = qual[col].replace('\\N', np.nan)
    qual[col] = qual[col].apply(lambda t: float(t.split(':')[0])*60+float(t.split(':')[1]) if isinstance(t, str) and ':' in t else (float(t) if pd.notnull(t) else np.nan))

# Use best qualifying time as a feature
qual['best_quali_time'] = qual[['q1','q2','q3']].min(axis=1)

# Rolling form: average qualifying position last 3 races
qual = qual.sort_values(['driverId','year','raceId'])
qual['qual_form_last3'] = qual.groupby('driverId')['position'].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())

# Features and targets
features = ['year','circuit','driver_name','team_name','best_quali_time','qual_form_last3']
cat_features = ['circuit','driver_name','team_name']
num_features = [f for f in features if f not in cat_features]

# Impute missing values
for col in num_features:
    qual[col] = qual[col].fillna(qual[col].median())
for col in cat_features:
    qual[col] = qual[col].fillna('Unknown')

# Encode categoricals
encoders = {}
for col in cat_features:
    le = LabelEncoder()
    qual[col] = le.fit_transform(qual[col].astype(str))
    encoders[col] = le

# Scale numerics
scaler = StandardScaler()
qual[num_features] = scaler.fit_transform(qual[num_features])

# Save encoders and scaler
os.makedirs('model', exist_ok=True)
joblib.dump(encoders, 'model/encoders_qualifying.pkl')
joblib.dump(scaler, 'model/scaler_qualifying.pkl')

# Prepare data
X = qual[features]
y_reg = qual['position'].astype(int)
y_class = (qual['position'].astype(int) <= 5).astype(int)  # Top 5 classification

# Train/test split
X_train, X_test, y_reg_train, y_reg_test, y_class_train, y_class_test = train_test_split(
    X, y_reg, y_class, test_size=0.2, random_state=42, stratify=y_class)

# Regression model
xgb_reg = xgb.XGBRegressor(objective='reg:squarederror')
xgb_reg.fit(X_train, y_reg_train)
y_reg_pred = xgb_reg.predict(X_test)
mae = mean_absolute_error(y_reg_test, y_reg_pred)
print(f"Qualifying Regression MAE: {mae:.3f}")

# Classification model
xgb_clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_clf.fit(X_train, y_class_train)
y_class_pred = xgb_clf.predict(X_test)
acc = accuracy_score(y_class_test, y_class_pred)
print(f"Qualifying Top 5 Classification Accuracy: {acc:.3f}")
print(classification_report(y_class_test, y_class_pred, target_names=['Not Top 5', 'Top 5']))

# Save the better model
if acc > (1 - mae / X_test.shape[0]):
    joblib.dump(xgb_clf, 'model/xgb_qualifying_top5.model')
    print("Saved classification model (top 5) as best.")
else:
    joblib.dump(xgb_reg, 'model/xgb_qualifying_regression.model')
    print("Saved regression model (position) as best.") 