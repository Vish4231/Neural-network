import fastf1
import pandas as pd
import numpy as np
from fastf1.ergast import Ergast
from collections import defaultdict

def get_race_data(year=2023):
    ergast = Ergast()
    content = ergast.get_race_results(season=year).content
    # Flatten the list of lists into a single list of dicts
    if isinstance(content, list) and len(content) > 0 and isinstance(content[0], list):
        flat_content = [item for sublist in content for item in sublist]
    else:
        flat_content = content
    # Debug: inspect structure
    print("Type of flat_content:", type(flat_content))
    print("Length of flat_content:", len(flat_content))
    print("First 3 elements:", flat_content[:3])
    print("Types of first 3 elements:", [type(x) for x in flat_content[:3]])
    # Concatenate DataFrames
    races = pd.concat(flat_content, ignore_index=True)
    races = races[races['positionOrder'].notna()]
    # Fetch qualifying results using FastF1
    quali_results = []
    for rnd in races['round'].unique():
        try:
            session = fastf1.get_session(year, int(rnd), 'Q')
            session.load()
            for drv in session.drivers:
                drv_laps = session.laps.pick_driver(drv)
                if not drv_laps.empty:
                    # Use the best qualifying lap position
                    quali_pos = drv_laps['Position'].min()
                    quali_results.append({'round': rnd, 'driver': drv, 'qualifying_position': quali_pos})
        except Exception as e:
            continue
    quali_df = pd.DataFrame(quali_results)
    # Merge race and quali on round and driver
    df = pd.merge(
        races,
        quali_df,
        left_on=['round', 'driverId'],
        right_on=['round', 'driver'],
        how='left'
    )
    df['finishing_position'] = df['positionOrder'].astype(int)
    df['constructor'] = df['constructorId']
    df['driver'] = df['driverId']
    df['race_name'] = df['raceName']
    df['season'] = df['season']
    df['race_round'] = df['round']
    # Weather and tire features
    weather_features = defaultdict(dict)
    tire_features = defaultdict(dict)
    for rnd in df['race_round'].unique():
        try:
            session = fastf1.get_session(year, int(rnd), 'R')
            session.load(telemetry=False, weather=True, laps=True)
            weather = session.weather_data
            if weather is not None and not weather.empty:
                weather_features[rnd]['air_temp'] = weather['AirTemp'].mean()
                weather_features[rnd]['humidity'] = weather['Humidity'].mean()
                weather_features[rnd]['rain'] = weather['Rainfall'].mean()
            else:
                weather_features[rnd]['air_temp'] = np.nan
                weather_features[rnd]['humidity'] = np.nan
                weather_features[rnd]['rain'] = np.nan
            laps = session.laps
            if laps is not None and not laps.empty:
                # Most common compound used in race by driver
                for drv in laps['Driver'].unique():
                    drv_laps = laps[laps['Driver'] == drv]
                    compound = drv_laps['Compound'].mode().iloc[0] if not drv_laps['Compound'].mode().empty else None
                    tire_features[(rnd, drv)] = compound
        except Exception as e:
            weather_features[rnd]['air_temp'] = np.nan
            weather_features[rnd]['humidity'] = np.nan
            weather_features[rnd]['rain'] = np.nan
    df['air_temp'] = df['race_round'].map(lambda x: weather_features[x]['air_temp'])
    df['humidity'] = df['race_round'].map(lambda x: weather_features[x]['humidity'])
    df['rain'] = df['race_round'].map(lambda x: weather_features[x]['rain'])
    df['tire_compound'] = df.apply(lambda row: tire_features.get((row['race_round'], row['driver']), None), axis=1)
    # Past results: average finish in last 3 races
    df = df.sort_values(['driver', 'race_round'])
    df['avg_finish_last3'] = df.groupby('driver')['finishing_position'].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    # Drop unnecessary columns
    df = df[['season', 'race_round', 'race_name', 'driver', 'constructor', 'qualifying_position', 'finishing_position', 'air_temp', 'humidity', 'rain', 'tire_compound', 'avg_finish_last3']]
    return df

if __name__ == '__main__':
    df = get_race_data(2023)
    print(df.head())
    df.to_csv('data/f1_race_data_2023_enriched.csv', index=False) 