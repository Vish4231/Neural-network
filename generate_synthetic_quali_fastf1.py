import fastf1
import pandas as pd
import numpy as np
import os

# Print FastF1 version to confirm update
print(f"Using FastF1 version: {fastf1.__version__}")

fastf1.Cache.enable_cache('fastf1_cache')

# Years to use for historical Spa qualifying data
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

# Fetch historical Spa qualifying data
qual_data = []
for year in YEARS:
    try:
        session = fastf1.get_session(year, SPA_EVENT_NAME, 'Q')
        session.load()
        laps = session.laps
        # Use FastF1's updated driver/team mapping if available
        for drv in laps['Driver'].unique():
            drv_laps = laps.pick_driver(drv)
            q1 = drv_laps.loc[drv_laps['SessionPart'] == 'Q1', 'LapTime'].min()
            q2 = drv_laps.loc[drv_laps['SessionPart'] == 'Q2', 'LapTime'].min()
            q3 = drv_laps.loc[drv_laps['SessionPart'] == 'Q3', 'LapTime'].min()
            # Use team name from FastF1 if available
            team_name = drv_laps['Team'].iloc[0] if 'Team' in drv_laps.columns and not drv_laps.empty else 'Unknown'
            qual_data.append({
                'year': year,
                'driver': drv,
                'team': team_name,
                'q1': q1.total_seconds() if pd.notnull(q1) else np.nan,
                'q2': q2.total_seconds() if pd.notnull(q2) else np.nan,
                'q3': q3.total_seconds() if pd.notnull(q3) else np.nan,
            })
    except Exception as e:
        print(f"Warning: Could not load Spa qualifying for {year}: {e}")

qual_df = pd.DataFrame(qual_data)

# Compute driver averages for Q1/Q2/Q3
driver_q_avgs = qual_df.groupby('driver').agg({'q1': 'mean', 'q2': 'mean', 'q3': 'mean'}).reset_index()

# Map 2025 drivers to historical averages (fallback to overall median if not found)
q1_median = driver_q_avgs['q1'].median()
q2_median = driver_q_avgs['q2'].median()
q3_median = driver_q_avgs['q3'].median()

synthetic_rows = []
for i, (drv, team) in enumerate(zip(drivers_2025, teams_2025)):
    drv_hist = driver_q_avgs[driver_q_avgs['driver'] == drv]
    q1 = drv_hist['q1'].values[0] if not drv_hist.empty else q1_median
    q2 = drv_hist['q2'].values[0] if not drv_hist.empty else q2_median
    q3 = drv_hist['q3'].values[0] if not drv_hist.empty else q3_median
    row = {
        'driver_name': drv,
        'team_name': team,
        'year': 2025,
        'circuit': 'circuit de spa francorchamps',
        'grid': i+1,  # Assign grid by order in lineup (can be randomized or sorted by q3)
        'q1': q1,
        'q2': q2,
        'q3': q3,
        'avg_lap_time': np.nanmean([q1, q2, q3]),
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

# Save to CSV for use in pipeline
synthetic_quali_df.to_csv('F1_2025_Dataset/Spa2025_SyntheticQuali_FastF1.csv', index=False)
print("Synthetic Spa 2025 qualifying data generated and saved as F1_2025_Dataset/Spa2025_SyntheticQuali_FastF1.csv") 