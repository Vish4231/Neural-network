import pandas as pd

# Paths to your data files
qual_path = "F1_2025_Dataset/F1_2025_QualifyingResults.csv"
race_path = "F1_2025_Dataset/F1_2025_RaceResults.csv"

# Load the data
qual_df = pd.read_csv(qual_path)
race_df = pd.read_csv(race_path)

# Find the order of races in 2025 and the index of Spa
race_order = list(race_df['Track'].unique())
spa_index = race_order.index("Spa-Francorchamps")

# Get all races before Spa
races_before_spa = race_order[:spa_index]

# Get all qualifying results before Spa
qual_2025 = qual_df[qual_df['Track'].isin(races_before_spa)]

# Only use numeric positions (ignore NC, DNF, etc.)
qual_2025_numeric = qual_2025[pd.to_numeric(qual_2025['Position'], errors='coerce').notnull()]
qual_2025_numeric['Position'] = qual_2025_numeric['Position'].astype(int)

# For each driver, get their average qualifying position in 2025 before Spa
driver_avg_qual = (
    qual_2025_numeric.groupby('Driver')['Position']
    .mean()
    .sort_values()
    .reset_index()
)

# For drivers with no prior qualifying, assign a high (bad) average
all_spa_drivers = race_df[race_df['Track'] == "Spa-Francorchamps"]['Driver'].unique()
for drv in all_spa_drivers:
    if drv not in driver_avg_qual['Driver'].values:
        driver_avg_qual = pd.concat([
            driver_avg_qual,
            pd.DataFrame({'Driver': [drv], 'Position': [21]})
        ], ignore_index=True)

# Sort by average qualifying position to simulate Spa grid
driver_avg_qual = driver_avg_qual.sort_values('Position').reset_index(drop=True)
driver_avg_qual['SimulatedGrid'] = range(1, len(driver_avg_qual) + 1)

# Update the Spa rows in the race results file
spa_mask = race_df['Track'] == "Spa-Francorchamps"
for _, row in driver_avg_qual.iterrows():
    race_df.loc[
        (spa_mask) & (race_df['Driver'] == row['Driver']),
        'Starting Grid'
    ] = row['SimulatedGrid']

# Save the updated race results
race_df.to_csv(race_path, index=False)
print("Updated Spa starting grid in F1_2025_RaceResults.csv based on 2025 qualifying form.")

# Print the simulated Spa qualifying order
print("\nSimulated Spa-Francorchamps 2025 Qualifying Order:")
print("Pos | Driver                | Team")
print("----+-----------------------+------------------------------")
spa_lineup = race_df[race_df['Track'] == "Spa-Francorchamps"][['Driver', 'Team', 'Starting Grid']]
spa_lineup = spa_lineup.merge(driver_avg_qual[['Driver', 'SimulatedGrid']], on='Driver')
spa_lineup = spa_lineup.sort_values('SimulatedGrid')
for _, row in spa_lineup.iterrows():
    print(f"{int(row['SimulatedGrid']):2d}  | {row['Driver']:<21} | {row['Team']}") 