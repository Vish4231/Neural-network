import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import requests
import pandas as pd
import os

def fetch_openf1_car_data(year=2023, limit=10000, save_path="data/car_data.csv"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    url = "https://api.openf1.org/v1/car_data"
    params = {
        "year": year,
        "limit": limit
    }
    print(f"Requesting: {url} with params {params}")
    response = requests.get(url, params=params)
    print(f"Status code: {response.status_code}")
    if response.status_code != 200:
        print(f"Error: Received status code {response.status_code}")
        print(f"Response text: {response.text}")
        return
    try:
        data = response.json()
    except Exception as e:
        print(f"Error decoding JSON: {e}")
        print(f"Response text: {response.text}")
        return
    if not data:
        print("Warning: No data returned from API.")
        return
    print(f"Sample data (first 2 records): {data[:2]}")
    df = pd.DataFrame(data)
    print(f"Columns: {df.columns.tolist()}")
    print(f"Number of records: {len(df)}")
    df.to_csv(save_path, index=False)
    print(f"Saved {len(df)} rows to {save_path}")

def fetch_openf1_race_results(year=2023, limit=10000, save_path="data/race_results_2023.csv"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    url = "https://api.openf1.org/v1/results"
    params = {
        "year": year,
        "limit": limit
    }
    print(f"Requesting: {url} with params {params}")
    response = requests.get(url, params=params)
    print(f"Status code: {response.status_code}")
    if response.status_code != 200:
        print(f"Error: Received status code {response.status_code}")
        print(f"Response text: {response.text}")
        return
    try:
        data = response.json()
    except Exception as e:
        print(f"Error decoding JSON: {e}")
        print(f"Response text: {response.text}")
        return
    if not data:
        print("Warning: No data returned from API.")
        return
    print(f"Sample data (first 2 records): {data[:2]}")
    df = pd.DataFrame(data)
    print(f"Columns: {df.columns.tolist()}")
    print(f"Number of records: {len(df)}")
    df.to_csv(save_path, index=False)
    print(f"Saved {len(df)} rows to {save_path}")


def load_and_prepare_data(path="data/race_results_2023.csv"):
    df = pd.read_csv(path)

    # Basic filtering and selection
    features = ['driver_number', 'team_name', 'circuit_short_name', 'weather', 'qualifying_position', 'tyre_compound']
    target = 'position'

    df = df.dropna(subset=features + [target])

    # Feature Engineering
    df['qualifying_position'] = df['qualifying_position'].astype(int)
    df['driver_skill'] = df.groupby('driver_number')['position'].transform(lambda x: 1 / (x.mean() + 1))
    df['team_performance'] = df.groupby('team_name')['position'].transform(lambda x: 1 / (x.mean() + 1))

    # Encode categorical variables
    cat_features = ['team_name', 'circuit_short_name', 'weather', 'tyre_compound']
    for col in cat_features:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    X = df[features + ['driver_skill', 'team_performance']]
    y = df[target].astype(int) - 1  # zero-based positions
    y = to_categorical(y, num_classes=20)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)

def build_and_train_model(X_train, X_test, y_train, y_test):
    model = Sequential()
    model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(20, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=25, batch_size=32, validation_data=(X_test, y_test))
    _, acc = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {acc * 100:.2f}%")

if __name__ == "__main__":
    fetch_openf1_race_results()
    X_train, X_test, y_train, y_test = load_and_prepare_data()
    build_and_train_model(X_train, X_test, y_train, y_test)