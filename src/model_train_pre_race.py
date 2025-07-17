import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Load data
DATA_PATH = 'data/pre_race_features.csv'
df = pd.read_csv(DATA_PATH)

# Drop rows with missing target or grid position
features = [
    'grid_position', 'qualifying_lap_time', 'air_temperature', 'humidity', 'rainfall',
    'track_temperature', 'wind_speed', 'team_name', 'driver_name', 'circuit', 'country_code'
]
target = 'finishing_position'
df = df.dropna(subset=features + [target])

# Filter out rows where finishing_position is not a valid integer (e.g., DQ, DNS, DNF)
def is_int_str(x):
    try:
        int(x)
        return True
    except:
        return False
invalid_rows = ~df[target].apply(is_int_str)
if invalid_rows.any():
    print(f"Dropping {invalid_rows.sum()} rows with non-integer finishing_position values: {df.loc[invalid_rows, target].unique().tolist()}")
    df = df[~invalid_rows]

# Encode categorical features
cat_features = ['team_name', 'driver_name', 'circuit', 'country_code']
encoders = {}
for col in cat_features:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

# Prepare X, y
X = df[features]
y = df[target].astype(int) - 1  # zero-based for Keras

# Scale numeric features
num_features = [f for f in features if f not in cat_features]
scaler = StandardScaler()
X[num_features] = scaler.fit_transform(X[num_features])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Build model
model = keras.Sequential([
    layers.Input(shape=(X.shape[1],)),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.2),
    layers.Dense(32, activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(20, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train
history = model.fit(X_train, y_train, epochs=40, batch_size=16, validation_data=(X_test, y_test))

# Evaluate
y_pred = np.argmax(model.predict(X_test), axis=1)
acc = accuracy_score(y_test, y_pred)
print(f"Test accuracy: {acc*100:.2f}%")
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, zero_division=0))

# Save model
os.makedirs('model', exist_ok=True)
model.save('model/pre_race_model.keras')
print("Model saved to model/pre_race_model.keras")

# Print sample predictions
sample = X_test.sample(5, random_state=42)
preds = np.argmax(model.predict(sample), axis=1) + 1
print("Sample predictions (grid_position, predicted_finish, true_finish):")
for i, idx in enumerate(sample.index):
    print(f"Grid: {int(df.loc[idx, 'grid_position'])}, Pred: {preds[i]}, True: {int(df.loc[idx, target])}") 