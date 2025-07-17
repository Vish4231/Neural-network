import requests
import pandas as pd
import os

def fetch_openf1_car_data(limit=10000, save_path="data/car_data.csv"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    url = "https://api.openf1.org/v1/car_data"
    params = {
        "limit": limit
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()
    df = pd.DataFrame(data)
    df.to_csv(save_path, index=False)
    print(f"Saved {len(df)} rows to {save_path}")

if __name__ == "__main__":
    fetch_openf1_car_data() 