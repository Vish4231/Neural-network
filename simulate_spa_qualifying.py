import pandas as pd
from typing import List, Tuple, Dict, Optional

def simulate_qualifying_grid(
    race_csv_path: str,
    qual_averages: List[Tuple[str, float]],
    track_name: str = "Spa-Francorchamps",
    grid_column: str = "Starting Grid",
    driver_column: str = "Driver",
    team_column: str = "Team",
    track_column: str = "Track",
    inplace: bool = False,
    output_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Simulate qualifying grid for a given track based on provided qualifying averages.
    """
    # Load race data
    race_df = pd.read_csv(race_csv_path)
    if track_column not in race_df.columns or driver_column not in race_df.columns:
        raise ValueError(f"Missing required columns: {track_column}, {driver_column}")

    # Filter for the specified track
    track_mask = race_df[track_column] == track_name
    track_drivers = race_df[track_mask][driver_column].tolist()

    # Build grid order based on averages
    avg_dict: Dict[str, float] = dict(qual_averages)
    missing_drivers = [drv for drv in track_drivers if drv not in avg_dict]
    if missing_drivers:
        print(f"Warning: No qualifying average for drivers: {missing_drivers}. Assigning to back of grid.")
    grid_order = sorted(
        [(drv, avg_dict.get(drv, 99)) for drv in track_drivers],
        key=lambda x: x[1]
    )

    # Assign grid positions
    for i, (drv, _) in enumerate(grid_order, 1):
        race_df.loc[(track_mask) & (race_df[driver_column] == drv), grid_column] = i

    # Ensure grid column is integer
    race_df[grid_column] = race_df[grid_column].astype(pd.Int64Dtype())

    # Optionally save
    if inplace or output_path:
        save_path = output_path if output_path else race_csv_path
        race_df.to_csv(save_path, index=False)
        print(f"Updated {track_name} starting grid in {save_path} based on provided qualifying averages.")

    # Print simulated grid
    print(f"\nSimulated {track_name} Qualifying Order:")
    print("Pos | Driver                | Team")
    print("----+-----------------------+------------------------------")
    lineup = race_df[track_mask][[driver_column, team_column, grid_column]].copy()
    lineup = lineup.sort_values(grid_column)
    for _, row in lineup.iterrows():
        print(f"{row[grid_column]:2d}  | {row[driver_column]:<21} | {row[team_column]}")

    return race_df

if __name__ == "__main__":
    # Silverstone 2025 qualifying results (user provided)
    qual_averages = [
        ("Max Verstappen", 1),
        ("Oscar Piastri", 2),
        ("Lando Norris", 3),
        ("George Russell", 4),
        ("Lewis Hamilton", 5),
        ("Charles Leclerc", 6),
        ("Andrea Kimi Antonelli", 7),
        ("Oliver Bearman", 8),
        ("Fernando Alonso", 9),
        ("Pierre Gasly", 10),
        ("Carlos Sainz", 11),
        ("Yuki Tsunoda", 12),
        ("Isack Hadjar", 13),
        ("Alexander Albon", 14),
        ("Esteban Ocon", 15),
        ("Liam Lawson", 16),
        ("Gabriel Bortoleto", 17),
        ("Lance Stroll", 18),
        ("Nico Hulkenberg", 19),
        ("Franco Colapinto", 20),
    ]
    simulate_qualifying_grid(
        race_csv_path="F1_2025_Dataset/F1_2025_RaceResults.csv",
        qual_averages=qual_averages,
        track_name="Silverstone",
        inplace=True
    )