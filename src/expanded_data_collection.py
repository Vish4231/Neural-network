#!/usr/bin/env python3
"""
Enhanced F1 Data Collection Script
Fetches comprehensive historical F1 data across multiple years with expanded features
"""

import requests
import pandas as pd
import time
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

class EnhancedF1DataCollector:
    def __init__(self, base_url="https://api.openf1.org/v1", output_dir="data"):
        self.base_url = base_url
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def fetch_with_retry(self, url: str, params: dict = None, max_retries: int = 3) -> Optional[dict]:
        """Fetch data with retry logic and rate limiting"""
        for attempt in range(max_retries):
            try:
                response = requests.get(url, params=params)
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:  # Rate limited
                    wait_time = 2 ** attempt
                    print(f"Rate limited, waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    print(f"HTTP {response.status_code}: {response.text}")
                    return None
            except Exception as e:
                print(f"Error fetching {url}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)
        return None
    
    def fetch_sessions(self, years: List[int] = None) -> pd.DataFrame:
        """Fetch all sessions for given years"""
        if years is None:
            years = list(range(2023, 2025))  # Default to 2023-2024
            
        all_sessions = []
        for year in years:
            print(f"Fetching sessions for {year}...")
            sessions = self.fetch_with_retry(f"{self.base_url}/sessions", {"year": year})
            if sessions:
                all_sessions.extend(sessions)
                print(f"Found {len(sessions)} sessions for {year}")
            time.sleep(0.5)  # Rate limiting
            
        df = pd.DataFrame(all_sessions)
        output_path = os.path.join(self.output_dir, "enhanced_sessions.csv")
        df.to_csv(output_path, index=False)
        print(f"Saved {len(df)} sessions to {output_path}")
        return df
    
    def fetch_race_results(self, years: List[int] = None) -> pd.DataFrame:
        """Fetch comprehensive race results"""
        if years is None:
            years = list(range(2023, 2025))
            
        all_results = []
        for year in years:
            print(f"Fetching race results for {year}...")
            results = self.fetch_with_retry(f"{self.base_url}/session_result", {"year": year})
            if results:
                all_results.extend(results)
                print(f"Found {len(results)} race results for {year}")
            time.sleep(0.5)
            
        df = pd.DataFrame(all_results)
        output_path = os.path.join(self.output_dir, "enhanced_race_results.csv")
        df.to_csv(output_path, index=False)
        print(f"Saved {len(df)} race results to {output_path}")
        return df
    
    def fetch_qualifying_results(self, years: List[int] = None) -> pd.DataFrame:
        """Fetch qualifying results for better grid position data"""
        if years is None:
            years = list(range(2023, 2025))
            
        all_quali = []
        for year in years:
            print(f"Fetching qualifying results for {year}...")
            # Get all qualifying sessions for the year
            sessions = self.fetch_with_retry(f"{self.base_url}/sessions", 
                                           {"year": year, "session_type": "Qualifying"})
            if sessions:
                for session in sessions:
                    session_key = session['session_key']
                    quali_results = self.fetch_with_retry(f"{self.base_url}/session_result", 
                                                        {"session_key": session_key})
                    if quali_results:
                        all_quali.extend(quali_results)
                    time.sleep(0.3)
                    
        df = pd.DataFrame(all_quali)
        output_path = os.path.join(self.output_dir, "enhanced_qualifying_results.csv")
        df.to_csv(output_path, index=False)
        print(f"Saved {len(df)} qualifying results to {output_path}")
        return df
    
    def fetch_practice_results(self, years: List[int] = None) -> pd.DataFrame:
        """Fetch practice session results for form analysis"""
        if years is None:
            years = list(range(2023, 2025))
            
        all_practice = []
        practice_types = ["Practice 1", "Practice 2", "Practice 3", "Sprint"]
        
        for year in years:
            print(f"Fetching practice results for {year}...")
            for practice_type in practice_types:
                sessions = self.fetch_with_retry(f"{self.base_url}/sessions", 
                                               {"year": year, "session_type": practice_type})
                if sessions:
                    for session in sessions:
                        session_key = session['session_key']
                        practice_results = self.fetch_with_retry(f"{self.base_url}/session_result", 
                                                              {"session_key": session_key})
                        if practice_results:
                            all_practice.extend(practice_results)
                        time.sleep(0.3)
                        
        df = pd.DataFrame(all_practice)
        output_path = os.path.join(self.output_dir, "enhanced_practice_results.csv")
        df.to_csv(output_path, index=False)
        print(f"Saved {len(df)} practice results to {output_path}")
        return df
    
    def fetch_driver_info(self, years: List[int] = None) -> pd.DataFrame:
        """Fetch comprehensive driver information"""
        if years is None:
            years = list(range(2023, 2025))
            
        all_drivers = []
        for year in years:
            print(f"Fetching driver info for {year}...")
            drivers = self.fetch_with_retry(f"{self.base_url}/drivers", {"year": year})
            if drivers:
                all_drivers.extend(drivers)
                print(f"Found {len(drivers)} driver records for {year}")
            time.sleep(0.5)
            
        df = pd.DataFrame(all_drivers)
        # Remove duplicates based on driver_number and year
        df = df.drop_duplicates(subset=['driver_number', 'session_key'])
        output_path = os.path.join(self.output_dir, "enhanced_driver_info.csv")
        df.to_csv(output_path, index=False)
        print(f"Saved {len(df)} driver records to {output_path}")
        return df
    
    def fetch_weather_data(self, years: List[int] = None) -> pd.DataFrame:
        """Fetch comprehensive weather data"""
        if years is None:
            years = list(range(2023, 2025))
            
        all_weather = []
        for year in years:
            print(f"Fetching weather data for {year}...")
            weather = self.fetch_with_retry(f"{self.base_url}/weather", {"year": year})
            if weather:
                all_weather.extend(weather)
                print(f"Found {len(weather)} weather records for {year}")
            time.sleep(0.5)
            
        df = pd.DataFrame(all_weather)
        output_path = os.path.join(self.output_dir, "enhanced_weather_data.csv")
        df.to_csv(output_path, index=False)
        print(f"Saved {len(df)} weather records to {output_path}")
        return df
    
    def fetch_car_data_sample(self, years: List[int] = None, sample_size: int = 1000) -> pd.DataFrame:
        """Fetch sample car data (limited due to size)"""
        if years is None:
            years = list(range(2023, 2026))
            
        all_car_data = []
        for year in years:
            print(f"Fetching car data sample for {year}...")
            car_data = self.fetch_with_retry(f"{self.base_url}/car_data", 
                                           {"year": year, "limit": sample_size})
            if car_data:
                all_car_data.extend(car_data)
                print(f"Found {len(car_data)} car data records for {year}")
            time.sleep(0.5)
            
        df = pd.DataFrame(all_car_data)
        output_path = os.path.join(self.output_dir, "enhanced_car_data_sample.csv")
        df.to_csv(output_path, index=False)
        print(f"Saved {len(df)} car data records to {output_path}")
        return df
    
    def fetch_pit_stops(self, years: List[int] = None) -> pd.DataFrame:
        """Fetch pit stop data"""
        if years is None:
            years = list(range(2023, 2025))
            
        all_pit_stops = []
        for year in years:
            print(f"Fetching pit stop data for {year}...")
            pit_stops = self.fetch_with_retry(f"{self.base_url}/pit", {"year": year})
            if pit_stops:
                all_pit_stops.extend(pit_stops)
                print(f"Found {len(pit_stops)} pit stop records for {year}")
            time.sleep(0.5)
            
        df = pd.DataFrame(all_pit_stops)
        output_path = os.path.join(self.output_dir, "enhanced_pit_stops.csv")
        df.to_csv(output_path, index=False)
        print(f"Saved {len(df)} pit stop records to {output_path}")
        return df
    
    def fetch_intervals(self, years: List[int] = None) -> pd.DataFrame:
        """Fetch timing intervals data"""
        if years is None:
            years = list(range(2023, 2025))
            
        all_intervals = []
        for year in years:
            print(f"Fetching intervals data for {year}...")
            intervals = self.fetch_with_retry(f"{self.base_url}/intervals", 
                                            {"year": year, "limit": 10000})
            if intervals:
                all_intervals.extend(intervals)
                print(f"Found {len(intervals)} interval records for {year}")
            time.sleep(0.5)
            
        df = pd.DataFrame(all_intervals)
        output_path = os.path.join(self.output_dir, "enhanced_intervals.csv")
        df.to_csv(output_path, index=False)
        print(f"Saved {len(df)} interval records to {output_path}")
        return df
    
    def create_circuit_characteristics(self) -> pd.DataFrame:
        """Create circuit characteristics dataset"""
        # This would ideally be expanded with real circuit data
        circuit_data = {
            'circuit_short_name': [
                'Bahrain', 'Saudi Arabia', 'Australia', 'Azerbaijan', 'Miami', 
                'Imola', 'Monaco', 'Spain', 'Canada', 'Austria', 'United Kingdom',
                'Hungary', 'Belgium', 'Netherlands', 'Italy', 'Singapore', 
                'Japan', 'Qatar', 'United States', 'Mexico', 'Brazil', 'Las Vegas', 'Abu Dhabi'
            ],
            'track_type': [
                'permanent', 'street', 'permanent', 'street', 'permanent',
                'permanent', 'street', 'permanent', 'permanent', 'permanent', 'permanent',
                'permanent', 'permanent', 'permanent', 'permanent', 'street',
                'permanent', 'permanent', 'permanent', 'permanent', 'permanent', 'street', 'permanent'
            ],
            'overtaking_difficulty': [
                2, 4, 3, 3, 2, 
                4, 5, 3, 2, 2, 3,
                4, 2, 4, 3, 3,
                3, 2, 3, 3, 2, 2, 3
            ],
            'avg_lap_time_sec': [
                95, 90, 85, 105, 90,
                85, 75, 80, 75, 70, 90,
                80, 105, 75, 85, 105,
                90, 85, 95, 80, 70, 85, 95
            ],
            'drs_zones': [
                3, 3, 3, 2, 3,
                2, 1, 2, 2, 3, 2,
                2, 3, 2, 2, 3,
                2, 2, 2, 3, 2, 2, 2
            ]
        }
        
        df = pd.DataFrame(circuit_data)
        output_path = os.path.join(self.output_dir, "circuit_characteristics.csv")
        df.to_csv(output_path, index=False)
        print(f"Created circuit characteristics dataset with {len(df)} circuits")
        return df
    
    def collect_all_data(self, years: List[int] = None) -> Dict[str, pd.DataFrame]:
        """Collect all available data"""
        if years is None:
            years = list(range(2023, 2025))
            
        print(f"Starting comprehensive data collection for years: {years}")
        print("="*50)
        
        datasets = {}
        
        # Core race data
        print("\n1. Fetching sessions...")
        datasets['sessions'] = self.fetch_sessions(years)
        
        print("\n2. Fetching race results...")
        datasets['race_results'] = self.fetch_race_results(years)
        
        print("\n3. Fetching qualifying results...")
        datasets['qualifying_results'] = self.fetch_qualifying_results(years)
        
        print("\n4. Fetching practice results...")
        datasets['practice_results'] = self.fetch_practice_results(years)
        
        print("\n5. Fetching driver information...")
        datasets['driver_info'] = self.fetch_driver_info(years)
        
        print("\n6. Fetching weather data...")
        datasets['weather_data'] = self.fetch_weather_data(years)
        
        print("\n7. Fetching pit stop data...")
        datasets['pit_stops'] = self.fetch_pit_stops(years)
        
        print("\n8. Fetching intervals data...")
        datasets['intervals'] = self.fetch_intervals(years)
        
        print("\n9. Fetching car data sample...")
        datasets['car_data'] = self.fetch_car_data_sample(years)
        
        print("\n10. Creating circuit characteristics...")
        datasets['circuit_characteristics'] = self.create_circuit_characteristics()
        
        # Save summary
        summary = {
            'collection_date': datetime.now().isoformat(),
            'years_collected': years,
            'datasets': {name: len(df) for name, df in datasets.items()}
        }
        
        with open(os.path.join(self.output_dir, "collection_summary.json"), 'w') as f:
            json.dump(summary, f, indent=2)
        
        print("\n" + "="*50)
        print("Data collection completed!")
        print("Summary:")
        for name, count in summary['datasets'].items():
            print(f"  {name}: {count:,} records")
        
        return datasets

def main():
    """Main function to run the enhanced data collection"""
    collector = EnhancedF1DataCollector()
    
    # Collect data for multiple years
    years_to_collect = [2023, 2024, 2025]  # Add more years as needed
    
    print("Enhanced F1 Data Collection Tool")
    print("="*40)
    print(f"Target years: {years_to_collect}")
    print(f"Output directory: {collector.output_dir}")
    print("\nStarting collection...")
    
    datasets = collector.collect_all_data(years_to_collect)
    
    print(f"\nAll data saved to {collector.output_dir}/")
    print("You can now use this enhanced dataset for training your neural network!")

if __name__ == "__main__":
    main()
