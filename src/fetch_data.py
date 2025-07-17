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

if __name__ == "__main__":
    fetch_openf1_race_results() 