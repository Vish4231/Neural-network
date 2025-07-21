import csv
import os

def debug_csv(file_path):
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader, 1):
            if i == 6:
                print(f"Line 6: {row}")
            if i <= 10:  # Print the first 10 lines
                print(f"Line {i}: {row}")
            if len(row) != 18:  # Assuming 18 is the expected number of fields based on your data shape
                print(f"Line {i} has {len(row)} fields: {row}")
            if i >= 20:  # Stop after checking the first 20 lines
                break

# Find the 2025 data CSV file
data_dir = '/Users/vishvasshiyam/Documents/Neural-network/F1_2025_Dataset'
csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]

if csv_files:
    csv_path = os.path.join(data_dir, csv_files[0])
    print(f"Analyzing file: {csv_path}")
    debug_csv(csv_path)
else:
    print("No CSV files found in the F1_2025_Dataset directory.")
