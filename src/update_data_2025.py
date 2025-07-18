#!/usr/bin/env python3
"""
Update F1 Dataset with 2024 and 2025 data
"""

import requests
import pandas as pd
import os
from datetime import datetime
import time

def fetch_all_years_data():
    """Fetch comprehensive data for all years including 2024 and 2025"""
    
    print("Fetching comprehensive F1 data for 2018-2025...")
    
    all_data = []
    
    # Years to fetch
    years = list(range(2018, 2026))  # 2018-2025
    
    for year in years:
        print(f"\nProcessing year {year}...")
        
        # Get all race sessions for this year
        sessions_url = f"https://api.openf1.org/v1/sessions?year={year}&session_type=Race"
        sessions_resp = requests.get(sessions_url)
        
        if sessions_resp.status_code != 200:
            print(f"  No sessions found for {year}")
            continue
            
        sessions = sessions_resp.json()
        print(f"  Found {len(sessions)} race sessions")
        
        year_data = []
        
        for session in sessions:
            session_key = session['session_key']
            meeting_key = session['meeting_key']
            circuit = session.get('circuit_short_name', 'Unknown')
            country = session.get('country_name', 'Unknown')
            
            # Get starting grid
            grid_url = f"https://api.openf1.org/v1/starting_grid?session_key={session_key}"
            grid_resp = requests.get(grid_url)
            
            if grid_resp.status_code != 200:
                continue
                
            grid_data = grid_resp.json()
            
            # Get session results
            results_url = f"https://api.openf1.org/v1/session_result?session_key={session_key}"
            results_resp = requests.get(results_url)
            
            if results_resp.status_code != 200:
                continue
                
            results_data = results_resp.json()
            
            # Create position mapping
            position_map = {}
            for result in results_data:
                position_map[result['driver_number']] = result['position']
            
            # Get driver info
            drivers_url = f"https://api.openf1.org/v1/drivers?session_key={session_key}"
            drivers_resp = requests.get(drivers_url)
            
            driver_map = {}
            if drivers_resp.status_code == 200:
                drivers_data = drivers_resp.json()
                for driver in drivers_data:
                    driver_map[driver['driver_number']] = driver
            
            # Get weather data
            weather_url = f"https://api.openf1.org/v1/weather?session_key={session_key}"
            weather_resp = requests.get(weather_url)
            
            weather_data = None
            if weather_resp.status_code == 200:
                weather_raw = weather_resp.json()
                if weather_raw:
                    weather_df = pd.DataFrame(weather_raw)
                    weather_data = {
                        'air_temperature': weather_df['air_temperature'].mean() if 'air_temperature' in weather_df else None,
                        'humidity': weather_df['humidity'].mean() if 'humidity' in weather_df else None,
                        'rainfall': weather_df['rainfall'].mean() if 'rainfall' in weather_df else None,
                        'track_temperature': weather_df['track_temperature'].mean() if 'track_temperature' in weather_df else None,
                        'wind_speed': weather_df['wind_speed'].mean() if 'wind_speed' in weather_df else None,
                    }
            
            # Process each grid position
            for grid_entry in grid_data:
                driver_number = grid_entry['driver_number']
                
                # Get driver info
                driver_info = driver_map.get(driver_number, {})
                
                # Create data row
                row = {
                    'year': year,
                    'circuit': circuit,
                    'country': country,
                    'driver_number': driver_number,
                    'grid_position': grid_entry['position'],
                    'qualifying_lap_time': grid_entry.get('lap_duration', None),
                    'finishing_position': position_map.get(driver_number, None),
                    'team_name': driver_info.get('team_name', None),
                    'driver_name': driver_info.get('full_name', None),
                    'country_code': driver_info.get('country_code', None),
                }
                
                # Add weather data
                if weather_data:
                    row.update(weather_data)
                
                year_data.append(row)
                
            time.sleep(0.1)  # Rate limiting
            
        all_data.extend(year_data)
        print(f"  Collected {len(year_data)} data points for {year}")
        
        # Short pause between years
        time.sleep(0.5)
    
    # Convert to DataFrame
    df = pd.DataFrame(all_data)
    
    # Filter out rows without essential data
    df = df.dropna(subset=['finishing_position', 'grid_position', 'team_name'])
    
    # Convert finishing_position to int and filter valid positions
    def is_valid_position(pos):
        try:
            int_pos = int(pos)
            return 1 <= int_pos <= 25  # Valid F1 positions
        except:
            return False
    
    df = df[df['finishing_position'].apply(is_valid_position)]
    df['finishing_position'] = df['finishing_position'].astype(int)
    df['finishing_position_int'] = df['finishing_position']
    
    # Sort by driver and year/circuit for form calculations
    df = df.sort_values(['driver_number', 'year', 'circuit'])
    
    # Calculate driver form (average finish in last 3 races)
    df['driver_form_last3'] = (
        df.groupby('driver_number')['finishing_position_int']
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    )
    
    # Calculate team form (average finish in last 3 races)
    df['team_form_last3'] = (
        df.groupby('team_name')['finishing_position_int']
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    )
    
    # Add qualifying gap to pole
    df['qualifying_gap_to_pole'] = None
    for (year, circuit), group in df.groupby(['year', 'circuit']):
        if 'qualifying_lap_time' in group.columns and not group['qualifying_lap_time'].isna().all():
            try:
                pole_time = group['qualifying_lap_time'].min()
                mask = (df['year'] == year) & (df['circuit'] == circuit)
                df.loc[mask, 'qualifying_gap_to_pole'] = df.loc[mask, 'qualifying_lap_time'] - pole_time
            except:
                pass
    
    # Add teammate grid delta
    df['teammate_grid_delta'] = None
    for (year, circuit, team), group in df.groupby(['year', 'circuit', 'team_name']):
        if len(group) == 2:  # Teams have 2 drivers
            drivers = group.sort_values('grid_position')
            if len(drivers) == 2:
                driver1_idx = drivers.index[0]
                driver2_idx = drivers.index[1]
                
                grid1 = drivers.iloc[0]['grid_position']
                grid2 = drivers.iloc[1]['grid_position']
                
                df.at[driver1_idx, 'teammate_grid_delta'] = grid1 - grid2
                df.at[driver2_idx, 'teammate_grid_delta'] = grid2 - grid1
    
    # Add static circuit characteristics
    circuit_characteristics = {
        'Baku': {'track_type': 'street', 'overtaking_difficulty': 4},
        'Jeddah': {'track_type': 'street', 'overtaking_difficulty': 5},
        'Melbourne': {'track_type': 'permanent', 'overtaking_difficulty': 3},
        'Sakhir': {'track_type': 'permanent', 'overtaking_difficulty': 3},
        'Miami': {'track_type': 'permanent', 'overtaking_difficulty': 3},
        'Imola': {'track_type': 'permanent', 'overtaking_difficulty': 3},
        'Monte Carlo': {'track_type': 'permanent', 'overtaking_difficulty': 3},
        'Catalunya': {'track_type': 'permanent', 'overtaking_difficulty': 3},
        'Montreal': {'track_type': 'permanent', 'overtaking_difficulty': 3},
        'Spielberg': {'track_type': 'permanent', 'overtaking_difficulty': 3},
        'Silverstone': {'track_type': 'permanent', 'overtaking_difficulty': 3},
        'Hungaroring': {'track_type': 'permanent', 'overtaking_difficulty': 4},
        'Spa': {'track_type': 'permanent', 'overtaking_difficulty': 2},
        'Zandvoort': {'track_type': 'permanent', 'overtaking_difficulty': 3},
        'Monza': {'track_type': 'permanent', 'overtaking_difficulty': 3},
        'Singapore': {'track_type': 'street', 'overtaking_difficulty': 2},
        'Suzuka': {'track_type': 'permanent', 'overtaking_difficulty': 3},
        'Lusail': {'track_type': 'permanent', 'overtaking_difficulty': 3},
        'Austin': {'track_type': 'permanent', 'overtaking_difficulty': 3},
        'Mexico City': {'track_type': 'permanent', 'overtaking_difficulty': 3},
        'Interlagos': {'track_type': 'permanent', 'overtaking_difficulty': 3},
        'Las Vegas': {'track_type': 'permanent', 'overtaking_difficulty': 3},
        'Yas Marina': {'track_type': 'permanent', 'overtaking_difficulty': 3},
        'Shanghai': {'track_type': 'permanent', 'overtaking_difficulty': 3},
    }
    
    df['track_type'] = df['circuit'].map(lambda x: circuit_characteristics.get(x, {}).get('track_type', 'permanent'))
    df['overtaking_difficulty'] = df['circuit'].map(lambda x: circuit_characteristics.get(x, {}).get('overtaking_difficulty', 3))
    
    # Add placeholder championship data
    df['driver_championship_position'] = None
    df['team_championship_position'] = None
    df['driver_points_season'] = None
    df['team_points_season'] = None
    df['weather_rain_forecast'] = None
    
    print(f"\nFinal dataset:")
    print(f"Total rows: {len(df)}")
    print(f"Years: {sorted(df['year'].unique())}")
    print(f"Circuits: {len(df['circuit'].unique())}")
    
    # Show year breakdown
    for year in sorted(df['year'].unique()):
        year_count = len(df[df['year'] == year])
        print(f"  {year}: {year_count} data points")
    
    return df

def main():
    """Main function"""
    print("F1 Data Update Tool - Adding 2024 and 2025 support")
    print("=" * 50)
    
    # Create data directory
    os.makedirs('data', exist_ok=True)
    
    # Fetch all data
    df = fetch_all_years_data()
    
    # Save to CSV
    output_path = 'data/pre_race_features.csv'
    df.to_csv(output_path, index=False)
    
    print(f"\nData saved to {output_path}")
    print(f"Dataset now includes {len(df)} race entries")
    
    # Check for 2025 data specifically
    df_2025 = df[df['year'] == 2025]
    if len(df_2025) > 0:
        print(f"\n2025 data available: {len(df_2025)} entries")
        print("2025 circuits:", df_2025['circuit'].unique().tolist())
    else:
        print("\n2025 data: Not yet available (season hasn't started)")
        print("The system is ready to incorporate 2025 data when it becomes available")
    
    print("\nData collection complete!")

if __name__ == "__main__":
    main()
