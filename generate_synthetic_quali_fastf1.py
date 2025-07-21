import fastf1
import pandas as pd
import numpy as np
import os

# Print FastF1 version to confirm update
print(f"Using FastF1 version: {fastf1.__version__}")

fastf1.Cache.enable_cache('fastf1_cache')

YEARS = [2018, 2019, 2020, 2021, 2022, 2023]
SPA_EVENT_NAME = 'Belgian Grand Prix'

# Load 2025 Spa lineup from your race results file
lineup_path = 'F1_2025_Dataset/F1_2025_RaceResults.csv'
lineup_df = pd.read_csv(lineup_path)
spa_lineup = lineup_df[lineup_df['Track'].str.lower().str.replace('-', ' ').str.replace('_', ' ') == 'spa francorchamps']

# Build driver/team list for 2025
# If FastF1 provides updated team/driver mapping utilities, use them here
# (Otherwise, fallback to the names in the CSV)
drivers_2025 = spa_lineup['Driver'].tolist()
teams_2025 = spa_lineup['Team'].tolist()

# Fetch historical Spa qualifying data (best lap per driver)
qual_data = []
for year in YEARS:
    try:
        session = fastf1.get_session(year, SPA_EVENT_NAME, 'Q')
        session.load()
        laps = session.laps
        for drv in laps['Driver'].unique():
            drv_laps = laps[laps['Driver'] == drv]
            best_lap = drv_laps['LapTime'].min()
            team_name = drv_laps['Team'].iloc[0] if 'Team' in drv_laps.columns and not drv_laps.empty else 'Unknown'
            qual_data.append({
                'year': year,
                'driver': drv,
                'team': team_name,
                'q_best': best_lap.total_seconds() if pd.notnull(best_lap) else np.nan,
            })
    except Exception as e:
        print(f"Warning: Could not load Spa qualifying for {year}: {e}")

qual_df = pd.DataFrame(qual_data)

# Compute driver averages for q_best
driver_q_avgs = qual_df.groupby('driver').agg({'q_best': 'mean'}).reset_index()
qbest_median = driver_q_avgs['q_best'].median()

synthetic_rows = []
for i, (drv, team) in enumerate(zip(drivers_2025, teams_2025)):
    drv_hist = driver_q_avgs[driver_q_avgs['driver'] == drv]
    q_best = drv_hist['q_best'].values[0] if not drv_hist.empty else qbest_median
    row = {
        'driver_name': drv,
        'team_name': team,
        'year': 2025,
        'circuit': 'circuit de spa francorchamps',
        'grid': i+1,  # Assign grid by order in lineup (can be randomized or sorted by q_best)
        'q_best': q_best,
        'avg_lap_time': q_best,
        'pit_stop_count': 0,
        'driver_form_last3': 10,
        'driver_form_last5': 10,
        'team_form_last3': 10,
        'team_form_last5': 10,
        'grid_vs_qual': 0,
        'pit_lap_interaction': 0,
    }
    synthetic_rows.append(row)

synthetic_quali_df = pd.DataFrame(synthetic_rows)
synthetic_quali_df.to_csv('F1_2025_Dataset/Spa2025_SyntheticQuali_FastF1.csv', index=False)
print("Synthetic Spa 2025 qualifying data generated and saved as F1_2025_Dataset/Spa2025_SyntheticQuali_FastF1.csv") 