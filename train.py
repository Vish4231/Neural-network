import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from src.model import build_model

# Load data
df = pd.read_csv('data/f1_race_data_2023_enriched.csv')

# Encode categorical features
le_driver = LabelEncoder()
le_constructor = LabelEncoder()
le_tire = LabelEncoder()

df['driver_enc'] = le_driver.fit_transform(df['driver'])
df['constructor_enc'] = le_constructor.fit_transform(df['constructor'])
df['tire_enc'] = le_tire.fit_transform(df['tire_compound'].fillna('Unknown'))

# Features and target
feature_cols = [
    'qualifying_position', 'constructor_enc', 'driver_enc', 'race_round',
    'air_temp', 'humidity', 'rain', 'avg_finish_last3'
]
X_num = df[feature_cols].values
X_tire = df['tire_enc'].values
y = df['finishing_position'].values - 1  # 0-indexed for softmax

# Scale numerical features
scaler = StandardScaler()
X_num = scaler.fit_transform(X_num)

# Train/test split
X_num_train, X_num_test, X_tire_train, X_tire_test, y_train, y_test = train_test_split(
    X_num, X_tire, y, test_size=0.2, random_state=42
)

# Build and train model
model = build_model(input_dim=X_num.shape[1]+1, tire_compound_vocab_size=len(le_tire.classes_), num_classes=20)
model.fit([X_num_train, X_tire_train], y_train, epochs=30, batch_size=16, validation_split=0.1)

# Evaluate
results = model.evaluate([X_num_test, X_tire_test], y_test)
print('Test results:', results) 