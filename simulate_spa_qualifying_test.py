import pandas as pd
import os
from simulate_spa_qualifying import simulate_qualifying_grid

def test_simulate_qualifying_grid():
    # Prepare test data
    test_data = {
        "Track": ["Spa-Francorchamps"] * 3 + ["Silverstone"],
        "Driver": ["Driver A", "Driver B", "Driver C", "Driver D"],
        "Team": ["Team X", "Team Y", "Team Z", "Team W"],
        "Starting Grid": [0, 0, 0, 0]
    }
    test_df = pd.DataFrame(test_data)
    test_csv = "test_race_results.csv"
    test_df.to_csv(test_csv, index=False)

    # Qualifying averages with missing driver
    qual_averages = [("Driver A", 2), ("Driver B", 1)]

    # Run simulation
    updated_df = simulate_qualifying_grid(
        race_csv_path=test_csv,
        qual_averages=qual_averages,
        track_name="Spa-Francorchamps",
        driver_column="Driver",
        team_column="Team",
        track_column="Track",
        grid_column="Starting Grid",
        inplace=False
    )

    # Check grid assignment
    spa_grid = updated_df[updated_df["Track"] == "Spa-Francorchamps"][["Driver", "Starting Grid"]]
    assert spa_grid.loc[spa_grid["Driver"] == "Driver B", "Starting Grid"].values[0] == 1
    assert spa_grid.loc[spa_grid["Driver"] == "Driver A", "Starting Grid"].values[0] == 2
    # Driver C missing in qual_averages should be assigned to back (default 99)
    assert spa_grid.loc[spa_grid["Driver"] == "Driver C", "Starting Grid"].values[0] == 3

    # Clean up
    os.remove(test_csv)
    print("All tests passed.")

if __name__ == "__main__":
    test_simulate_qualifying_grid()
