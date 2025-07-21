import pandas as pd
import os
import argparse
import fastf1
from fastf1.ergast import Ergast
from datetime import datetime

def fetch_and_append_new_results(year=2025):
    """
    Fetches the latest completed race results for a given year from the Ergast API,
    standardizes them, and appends them to the historical races and results files.
    """
    print(f"--- Starting Live Data Update for {year} ---")

    # --- Configuration ---
    ARCHIVE_PATH = 'archive (1)/'
    
    # --- Load Existing Data ---
    try:
        races_df = pd.read_csv(os.path.join(ARCHIVE_PATH, 'races.csv'))
        results_df = pd.read_csv(os.path.join(ARCHIVE_PATH, 'results.csv'))
        drivers_df = pd.read_csv(os.path.join(ARCHIVE_PATH, 'drivers.csv'))
        constructors_df = pd.read_csv(os.path.join(ARCHIVE_PATH, 'constructors.csv'))
        circuits_df = pd.read_csv(os.path.join(ARCHIVE_PATH, 'circuits.csv'))
        print("Loaded all historical data files.")
    except FileNotFoundError as e:
        print(f"Error: Could not find a required data file: {e}")
        return

    # --- Fetch New Data ---
    ergast = Ergast()
    
    # 1. Fetch the schedule for the target year using fastf1
    print(f"Fetching race schedule for {year}...")
    try:
        fastf1.Cache.enable_cache(os.path.join(os.getcwd(), 'cache'))
        season_schedule = fastf1.get_event_schedule(year, include_testing=False)
    except Exception as e:
        print(f"Could not fetch event schedule for {year}: {e}")
        return
    
    if season_schedule.empty:
        print(f"No race schedule found for {year}.")
        return

    # Create a mapping from circuit name to circuitId for later use
    circuit_name_to_id = circuits_df.set_index('name')['circuitId']
    
    existing_races_in_year = races_df[races_df['year'] == year]
    new_results_to_append = []
    new_races_to_append = []

    # 2. Iterate through each race in the schedule
    for _, event in season_schedule.iterrows():
        race_name = event['EventName']
        race_date = event['EventDate']

        # 3. Check if the race has happened and is not already in our data
        if race_date > pd.Timestamp.now(tz='UTC') or race_name in existing_races_in_year['name'].values:
            if race_date <= pd.Timestamp.now(tz='UTC'):
                print(f"Skipping '{race_name}' (already in dataset).")
            continue

        print(f"Found new completed race: '{race_name}'. Fetching results...")
        
        # 4. Fetch results for this specific race using Ergast
        race_results_data = ergast.get_race_results(season=year, round=event['RoundNumber'])
        if not race_results_data.content:
            print(f"No results found for '{race_name}'. Skipping.")
            continue
        
        race_result_df = race_results_data.content[0]

        # 5. Add the new race to our list of races to be added
        last_race_id = races_df['raceId'].max() if not new_races_to_append else max(r['raceId'] for r in new_races_to_append)
        new_race_id = last_race_id + 1
        
        circuit_id = circuit_name_to_id.get(event['Location']) # Match on location name

        new_races_to_append.append({
            'raceId': new_race_id,
            'year': year,
            'round': event['RoundNumber'],
            'circuitId': circuit_id,
            'name': race_name,
            'date': race_date.strftime('%Y-%m-%d'),
        })

        # 6. Process the results for this new race
        for _, row in race_result_df.iterrows():
            try:
                driver_id = drivers_df.loc[drivers_df['driverRef'] == row['driverId'], 'driverId'].iloc[0]
                constructor_id = constructors_df.loc[constructors_df['constructorRef'] == row['constructorId'], 'constructorId'].iloc[0]
            except IndexError:
                print(f"Warning: Could not map driver '{row['driverId']}' or constructor '{row['constructorId']}'. Skipping row.")
                continue

            new_results_to_append.append({
                'resultId': None,
                'raceId': new_race_id,
                'driverId': driver_id,
                'constructorId': constructor_id,
                'number': row['number'],
                'grid': row['grid'],
                'position': row.get('position', None),
                'positionText': str(row.get('positionText', '')),
                'positionOrder': row['position'],
                'points': row['points'],
                'laps': row['laps'],
                'time': row.get('time', None),
                'milliseconds': row.get('milliseconds', None),
                'fastestLap': row.get('fastestLap', None),
                'rank': row.get('rank', None),
                'fastestLapTime': row.get('fastestLapTime', None),
                'fastestLapSpeed': row.get('fastestLapSpeed', None),
                'statusId': 1 # Simplified status
            })

    # --- Append New Data if any was found ---
    if not new_results_to_append:
        print("No new, unprocessed race results found.")
        return

    # Append races
    new_races_df = pd.DataFrame(new_races_to_append)
    updated_races = pd.concat([races_df, new_races_df], ignore_index=True)
    
    # Append results
    new_results_df = pd.DataFrame(new_results_to_append)
    last_result_id = results_df['resultId'].max()
    new_results_df['resultId'] = range(last_result_id + 1, last_result_id + 1 + len(new_results_df))
    updated_results = pd.concat([results_df, new_results_df], ignore_index=True)

    # --- Save Updated Files ---
    races_path = os.path.join(ARCHIVE_PATH, 'races.csv')
    results_path = os.path.join(ARCHIVE_PATH, 'results.csv')

    # Backup and save
    os.rename(races_path, races_path + '.bak')
    os.rename(results_path, results_path + '.bak')
    print("Created backups of original races and results files.")

    updated_races.to_csv(races_path, index=False)
    updated_results.to_csv(results_path, index=False)
    
    print(f"Successfully added {len(new_races_df)} new races and {len(new_results_df)} results.")
    print("--- Live Data Update Complete ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update historical F1 data with the latest season results from a live API.")
    parser.add_argument('--year', type=int, default=datetime.now().year, help='The year to fetch data for.')
    args = parser.parse_args()
    
    fetch_and_append_new_results(year=args.year)
