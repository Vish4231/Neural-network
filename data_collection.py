#!/usr/bin/env python3
"""
FastF1 Data Collection Script
Collects comprehensive F1 data from FastF1 API for model training
"""

import fastf1
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import warnings
import time
from typing import Dict, List, Optional, Tuple
import pickle

warnings.filterwarnings('ignore')

class F1DataCollector:
    def __init__(self, cache_dir: str = "fastf1_cache"):
        """Initialize FastF1 data collector"""
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # Enable FastF1 cache
        fastf1.Cache.enable_cache(cache_dir)
        
        print("🏎️ F1 Data Collector Initialized")
        print(f"📁 Cache directory: {cache_dir}")
        
    def collect_season_data(self, year: int) -> pd.DataFrame:
        """Collect all race data for a season"""
        print(f"\n📅 Collecting {year} season data...")
        
        # Get race schedule
        schedule = fastf1.get_event_schedule(year)
        race_data = []
        
        for idx, race in schedule.iterrows():
            race_name = race['EventName']
            
            # Skip non-race events
            if 'Testing' in race_name or 'Sprint' in race_name:
                continue
                
            print(f"🏁 Processing {race_name}...")
            
            try:
                # Load race session
                session = fastf1.get_session(year, race_name, 'Race')
                session.load()
                
                # Get race results
                results = session.results
                
                if results.empty:
                    print(f"⚠️ No results for {race_name}")
                    continue
                
                # Add race metadata
                results['Year'] = year
                results['RaceName'] = race_name
                results['EventName'] = race['EventName']
                results['Country'] = race['Country']
                results['Date'] = race['Session5Date'] if 'Session5Date' in race else race['Session4Date']
                
                # Get lap times and performance data
                lap_data = self._get_lap_performance(session, race_name)
                
                # Merge with results
                enhanced_results = self._enhance_results(results, lap_data)
                
                race_data.append(enhanced_results)
                
                # Small delay to avoid overloading the API
                time.sleep(1)
                
            except Exception as e:
                print(f"❌ Error processing {race_name}: {e}")
                continue
        
        if race_data:
            season_df = pd.concat(race_data, ignore_index=True)
            print(f"✅ Collected {len(season_df)} results for {year}")
            return season_df
        else:
            print(f"❌ No data collected for {year}")
            return pd.DataFrame()
    
    def _get_lap_performance(self, session, race_name: str) -> Dict:
        """Extract lap performance metrics"""
        try:
            laps = session.laps
            
            if laps.empty:
                return {}
            
            # Calculate performance metrics per driver
            performance = {}
            
            for driver in laps['Driver'].unique():
                driver_laps = laps[laps['Driver'] == driver]
                
                # Skip if no valid laps
                valid_laps = driver_laps[driver_laps['LapTime'].notna()]
                if len(valid_laps) == 0:
                    continue
                
                # Calculate key metrics
                metrics = {
                    'AvgLapTime': valid_laps['LapTime'].mean().total_seconds() if not valid_laps['LapTime'].empty else None,
                    'BestLapTime': valid_laps['LapTime'].min().total_seconds() if not valid_laps['LapTime'].empty else None,
                    'LapCount': len(valid_laps),
                    'Consistency': valid_laps['LapTime'].std().total_seconds() if len(valid_laps) > 1 else 0,
                }
                
                # Add sector times if available
                if 'Sector1Time' in valid_laps.columns:
                    sector1_times = valid_laps['Sector1Time'].dropna()
                    if not sector1_times.empty:
                        metrics['AvgSector1'] = sector1_times.mean().total_seconds()
                
                if 'Sector2Time' in valid_laps.columns:
                    sector2_times = valid_laps['Sector2Time'].dropna()
                    if not sector2_times.empty:
                        metrics['AvgSector2'] = sector2_times.mean().total_seconds()
                
                if 'Sector3Time' in valid_laps.columns:
                    sector3_times = valid_laps['Sector3Time'].dropna()
                    if not sector3_times.empty:
                        metrics['AvgSector3'] = sector3_times.mean().total_seconds()
                
                performance[driver] = metrics
            
            return performance
            
        except Exception as e:
            print(f"⚠️ Error getting lap performance for {race_name}: {e}")
            return {}
    
    def _enhance_results(self, results: pd.DataFrame, lap_data: Dict) -> pd.DataFrame:
        """Enhance results with lap performance data"""
        enhanced = results.copy()
        
        # Add performance metrics
        for idx, row in enhanced.iterrows():
            driver = row['Abbreviation'] if 'Abbreviation' in row else row.get('DriverId', '')
            
            if driver in lap_data:
                for metric, value in lap_data[driver].items():
                    enhanced.at[idx, metric] = value
            else:
                # Fill with NaN if no data
                enhanced.at[idx, 'AvgLapTime'] = None
                enhanced.at[idx, 'BestLapTime'] = None
                enhanced.at[idx, 'LapCount'] = 0
                enhanced.at[idx, 'Consistency'] = None
        
        return enhanced
    
    def collect_multi_year_data(self, years: List[int]) -> pd.DataFrame:
        """Collect data for multiple years"""
        print(f"🔄 Starting multi-year data collection for: {years}")
        
        all_data = []
        
        for year in years:
            try:
                year_data = self.collect_season_data(year)
                if not year_data.empty:
                    all_data.append(year_data)
            except Exception as e:
                print(f"❌ Failed to collect {year} data: {e}")
                continue
        
        if all_data:
            combined_data = pd.concat(all_data, ignore_index=True)
            print(f"✅ Combined dataset: {len(combined_data)} race results")
            return combined_data
        else:
            print("❌ No data collected")
            return pd.DataFrame()
    
    def save_data(self, data: pd.DataFrame, filename: str):
        """Save collected data"""
        os.makedirs('data', exist_ok=True)
        filepath = f"data/{filename}"
        
        # Save as both CSV and pickle for flexibility
        data.to_csv(f"{filepath}.csv", index=False)
        data.to_pickle(f"{filepath}.pkl")
        
        print(f"💾 Data saved to {filepath}.csv and {filepath}.pkl")
        print(f"📊 Dataset shape: {data.shape}")
    
    def load_data(self, filename: str) -> pd.DataFrame:
        """Load previously saved data"""
        filepath = f"data/{filename}.pkl"
        
        if os.path.exists(filepath):
            data = pd.read_pickle(filepath)
            print(f"📂 Loaded data from {filepath}")
            print(f"📊 Dataset shape: {data.shape}")
            return data
        else:
            print(f"❌ File not found: {filepath}")
            return pd.DataFrame()

def main():
    """Main data collection routine"""
    collector = F1DataCollector()
    
    # Collect recent years data (2022-2024 for training)
    years = [2022, 2023, 2024]
    
    print("🚀 Starting F1 data collection...")
    print("This may take several minutes due to API rate limiting...")
    
    # Collect data
    data = collector.collect_multi_year_data(years)
    
    if not data.empty:
        # Save the data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        collector.save_data(data, f"f1_race_data_{timestamp}")
        
        # Display basic statistics
        print("\n📊 Data Collection Summary:")
        print(f"Total races: {data['RaceName'].nunique()}")
        print(f"Total drivers: {data['FullName'].nunique()}")
        print(f"Total teams: {data['TeamName'].nunique()}")
        print(f"Years covered: {sorted(data['Year'].unique())}")
        
        # Show recent Mexico data if available
        mexico_data = data[data['RaceName'].str.contains('Mexico', na=False)]
        if not mexico_data.empty:
            print(f"\n🇲🇽 Mexico GP data found: {len(mexico_data)} results")
            print("Recent Mexico winners:")
            for year in sorted(mexico_data['Year'].unique(), reverse=True):
                winner = mexico_data[mexico_data['Year'] == year].iloc[0]
                print(f"  {year}: {winner['FullName']} ({winner['TeamName']})")
        
    else:
        print("❌ Data collection failed")

if __name__ == "__main__":
    main()