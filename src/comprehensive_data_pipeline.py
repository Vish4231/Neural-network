#!/usr/bin/env python3
"""
Comprehensive F1 Data Pipeline
Phase 1: Data Cleaning & Integration from FastF1 API and all available sources
"""

import pandas as pd
import numpy as np
import fastf1
import requests
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class ComprehensiveF1DataPipeline:
    def __init__(self, cache_dir='fastf1_cache', data_dir='data'):
        self.cache_dir = cache_dir
        self.data_dir = data_dir
        self.openf1_base_url = "https://api.openf1.org/v1"
        
        # Enable FastF1 cache
        fastf1.Cache.enable_cache(cache_dir)
        
        # Create directories
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(cache_dir, exist_ok=True)
        
        print("🔧 Comprehensive F1 Data Pipeline Initialized")
    
    def clean_fastf1_data(self, years: List[int] = None) -> pd.DataFrame:
        """
        Clean and standardize FastF1 data from multiple years
        """
        if years is None:
            years = list(range(2018, 2026))  # 2018-2025
        
        print(f"📊 Cleaning FastF1 data for years: {years}")
        
        all_race_data = []
        all_quali_data = []
        all_weather_data = []
        
        for year in years:
            try:
                print(f"  Processing {year}...")
                
                # Get race schedule
                schedule = fastf1.get_event_schedule(year, include_testing=False)
                
                for _, event in schedule.iterrows():
                    try:
                        # Race data
                        race_data = self._extract_race_session_data(year, event, 'Race')
                        if not race_data.empty:
                            all_race_data.append(race_data)
                        
                        # Qualifying data
                        quali_data = self._extract_race_session_data(year, event, 'Qualifying')
                        if not quali_data.empty:
                            all_quali_data.append(quali_data)
                        
                        # Weather data
                        weather_data = self._extract_weather_data(year, event)
                        if not weather_data.empty:
                            all_weather_data.append(weather_data)
                            
                    except Exception as e:
                        print(f"    Warning: Could not process {event['EventName']} {year}: {e}")
                        continue
                        
            except Exception as e:
                print(f"  Warning: Could not process year {year}: {e}")
                continue
        
        # Combine all data
        combined_data = self._combine_session_data(all_race_data, all_quali_data, all_weather_data)
        
        print(f"✅ FastF1 data cleaning complete: {len(combined_data)} records")
        return combined_data
    
    def _extract_race_session_data(self, year: int, event, session_type: str) -> pd.DataFrame:
        """Extract data from a specific session"""
        try:
            session = fastf1.get_session(year, event['RoundNumber'], session_type)
            session.load()
            
            if session.results is None or session.results.empty:
                return pd.DataFrame()
            
            # Extract basic results
            results_data = []
            for _, result in session.results.iterrows():
                row = {
                    'year': year,
                    'round': event['RoundNumber'],
                    'event_name': event['EventName'],
                    'location': event['Location'],
                    'session_type': session_type,
                    'driver_name': f"{result.get('FirstName', '')} {result.get('LastName', '')}".strip(),
                    'team_name': result.get('TeamName', ''),
                    'driver_number': result.get('DriverNumber', ''),
                    'position': result.get('Position', np.nan),
                    'points': result.get('Points', 0),
                    'grid_position': result.get('GridPosition', np.nan),
                    'status': result.get('Status', ''),
                    'fastest_lap': result.get('FastestLap', False),
                    'fastest_lap_time': result.get('FastestLapTime', np.nan),
                    'fastest_lap_speed': result.get('FastestLapSpeed', np.nan)
                }
                
                # Add lap data if available
                if session.laps is not None and not session.laps.empty:
                    driver_laps = session.laps[session.laps['Driver'] == result.get('Abbreviation', '')]
                    if not driver_laps.empty:
                        row.update({
                            'avg_lap_time': driver_laps['LapTime'].mean().total_seconds() if pd.notna(driver_laps['LapTime'].mean()) else np.nan,
                            'best_lap_time': driver_laps['LapTime'].min().total_seconds() if pd.notna(driver_laps['LapTime'].min()) else np.nan,
                            'total_laps': len(driver_laps),
                            'avg_speed': driver_laps['SpeedI1'].mean() if 'SpeedI1' in driver_laps.columns else np.nan,
                            'max_speed': driver_laps['SpeedI1'].max() if 'SpeedI1' in driver_laps.columns else np.nan
                        })
                
                results_data.append(row)
            
            return pd.DataFrame(results_data)
            
        except Exception as e:
            print(f"    Error extracting {session_type} data: {e}")
            return pd.DataFrame()
    
    def _extract_weather_data(self, year: int, event) -> pd.DataFrame:
        """Extract weather data for an event"""
        try:
            # Try to get weather from any available session
            for session_type in ['Race', 'Qualifying', 'Practice 3', 'Practice 2', 'Practice 1']:
                try:
                    session = fastf1.get_session(year, event['RoundNumber'], session_type)
                    session.load()
                    
                    if session.weather_data is not None and not session.weather_data.empty:
                        weather_summary = {
                            'year': year,
                            'round': event['RoundNumber'],
                            'event_name': event['EventName'],
                            'location': event['Location'],
                            'avg_air_temp': session.weather_data['AirTemp'].mean() if 'AirTemp' in session.weather_data.columns else np.nan,
                            'avg_track_temp': session.weather_data['TrackTemp'].mean() if 'TrackTemp' in session.weather_data.columns else np.nan,
                            'avg_humidity': session.weather_data['Humidity'].mean() if 'Humidity' in session.weather_data.columns else np.nan,
                            'avg_wind_speed': session.weather_data['WindSpeed'].mean() if 'WindSpeed' in session.weather_data.columns else np.nan,
                            'avg_wind_direction': session.weather_data['WindDirection'].mean() if 'WindDirection' in session.weather_data.columns else np.nan,
                            'rainfall': session.weather_data['Rainfall'].sum() if 'Rainfall' in session.weather_data.columns else 0,
                            'rain_probability': (session.weather_data['Rainfall'] > 0).mean() if 'Rainfall' in session.weather_data.columns else 0
                        }
                        return pd.DataFrame([weather_summary])
                except:
                    continue
            
            return pd.DataFrame()
            
        except Exception as e:
            print(f"    Error extracting weather data: {e}")
            return pd.DataFrame()
    
    def _combine_session_data(self, race_data: List[pd.DataFrame], 
                            quali_data: List[pd.DataFrame], 
                            weather_data: List[pd.DataFrame]) -> pd.DataFrame:
        """Combine all session data into a unified dataset"""
        
        # Combine race data
        if race_data:
            combined_race = pd.concat(race_data, ignore_index=True)
        else:
            combined_race = pd.DataFrame()
        
        # Combine qualifying data
        if quali_data:
            combined_quali = pd.concat(quali_data, ignore_index=True)
        else:
            combined_quali = pd.DataFrame()
        
        # Combine weather data
        if weather_data:
            combined_weather = pd.concat(weather_data, ignore_index=True)
        else:
            combined_weather = pd.DataFrame()
        
        # Merge race and qualifying data
        if not combined_race.empty and not combined_quali.empty:
            # Merge on year, round, driver_name, team_name
            merged = combined_race.merge(
                combined_quali[['year', 'round', 'driver_name', 'team_name', 'position', 'grid_position', 'avg_lap_time', 'best_lap_time']],
                on=['year', 'round', 'driver_name', 'team_name'],
                how='left',
                suffixes=('_race', '_quali')
            )
        else:
            merged = combined_race if not combined_race.empty else combined_quali
        
        # Add weather data
        if not merged.empty and not combined_weather.empty:
            merged = merged.merge(
                combined_weather,
                on=['year', 'round', 'event_name', 'location'],
                how='left'
            )
        
        return merged
    
    def clean_openf1_data(self, years: List[int] = None) -> pd.DataFrame:
        """
        Clean and standardize OpenF1 API data
        """
        if years is None:
            years = list(range(2023, 2026))  # OpenF1 has limited historical data
        
        print(f"🌐 Cleaning OpenF1 data for years: {years}")
        
        all_data = []
        
        for year in years:
            try:
                print(f"  Processing OpenF1 {year}...")
                
                # Get sessions
                sessions = self._fetch_openf1_sessions(year)
                if sessions.empty:
                    continue
                
                for _, session in sessions.iterrows():
                    try:
                        # Get race results
                        results = self._fetch_openf1_results(session['session_key'])
                        if not results.empty:
                            results['year'] = year
                            results['session_key'] = session['session_key']
                            results['circuit_key'] = session['circuit_key']
                            results['session_name'] = session['session_name']
                            all_data.append(results)
                        
                        # Get car data
                        car_data = self._fetch_openf1_car_data(session['session_key'])
                        if not car_data.empty:
                            car_data['year'] = year
                            car_data['session_key'] = session['session_key']
                            car_data['circuit_key'] = session['circuit_key']
                            all_data.append(car_data)
                            
                    except Exception as e:
                        print(f"    Warning: Could not process session {session['session_key']}: {e}")
                        continue
                        
            except Exception as e:
                print(f"  Warning: Could not process OpenF1 year {year}: {e}")
                continue
        
        if all_data:
            combined_data = pd.concat(all_data, ignore_index=True)
            print(f"✅ OpenF1 data cleaning complete: {len(combined_data)} records")
            return combined_data
        else:
            print("⚠️ No OpenF1 data found")
            return pd.DataFrame()
    
    def _fetch_openf1_sessions(self, year: int) -> pd.DataFrame:
        """Fetch sessions from OpenF1 API"""
        try:
            url = f"{self.openf1_base_url}/sessions"
            params = {'year': year}
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return pd.DataFrame(data)
            else:
                print(f"    HTTP {response.status_code} for sessions {year}")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"    Error fetching sessions for {year}: {e}")
            return pd.DataFrame()
    
    def _fetch_openf1_results(self, session_key: int) -> pd.DataFrame:
        """Fetch race results from OpenF1 API"""
        try:
            url = f"{self.openf1_base_url}/results"
            params = {'session_key': session_key}
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return pd.DataFrame(data)
            else:
                return pd.DataFrame()
                
        except Exception as e:
            return pd.DataFrame()
    
    def _fetch_openf1_car_data(self, session_key: int) -> pd.DataFrame:
        """Fetch car data from OpenF1 API"""
        try:
            url = f"{self.openf1_base_url}/car_data"
            params = {'session_key': session_key, 'limit': 1000}
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return pd.DataFrame(data)
            else:
                return pd.DataFrame()
                
        except Exception as e:
            return pd.DataFrame()
    
    def clean_historical_data(self) -> pd.DataFrame:
        """
        Clean and standardize historical F1 data from archive
        """
        print("📚 Cleaning historical F1 data from archive...")
        
        archive_path = 'archive (1)/'
        if not os.path.exists(archive_path):
            print("⚠️ Archive directory not found")
            return pd.DataFrame()
        
        try:
            # Load all CSV files
            results = pd.read_csv(archive_path + 'results.csv')
            races = pd.read_csv(archive_path + 'races.csv')
            drivers = pd.read_csv(archive_path + 'drivers.csv')
            constructors = pd.read_csv(archive_path + 'constructors.csv')
            qualifying = pd.read_csv(archive_path + 'qualifying.csv')
            pit_stops = pd.read_csv(archive_path + 'pit_stops.csv')
            lap_times = pd.read_csv(archive_path + 'lap_times.csv')
            
            # Merge data
            merged = results.merge(races, on='raceId', how='left')
            merged = merged.merge(drivers, on='driverId', how='left')
            merged = merged.merge(constructors, on='constructorId', how='left')
            
            # Add qualifying data
            merged = merged.merge(
                qualifying[['raceId', 'driverId', 'constructorId', 'q1', 'q2', 'q3', 'position']],
                on=['raceId', 'driverId', 'constructorId'],
                how='left',
                suffixes=('', '_qual')
            )
            
            # Add pit stop data
            pit_counts = pit_stops.groupby(['raceId', 'driverId']).size().reset_index(name='pit_stop_count')
            merged = merged.merge(pit_counts, on=['raceId', 'driverId'], how='left')
            merged['pit_stop_count'] = merged['pit_stop_count'].fillna(0)
            
            # Add lap time data
            lap_times['lap_time_sec'] = lap_times['milliseconds'] / 1000.0
            avg_lap = lap_times.groupby(['raceId', 'driverId'])['lap_time_sec'].mean().reset_index(name='avg_lap_time')
            merged = merged.merge(avg_lap, on=['raceId', 'driverId'], how='left')
            
            # Standardize column names
            merged = merged.rename(columns={
                'name': 'circuit',
                'name_team': 'team_name',
                'surname': 'driver_surname',
                'forename': 'driver_forename',
                'position': 'finishing_position',
                'position_qual': 'qualifying_position'
            })
            
            # Create driver name
            merged['driver_name'] = merged['driver_forename'] + ' ' + merged['driver_surname']
            
            print(f"✅ Historical data cleaning complete: {len(merged)} records")
            return merged
            
        except Exception as e:
            print(f"❌ Error cleaning historical data: {e}")
            return pd.DataFrame()
    
    def integrate_all_data_sources(self) -> pd.DataFrame:
        """
        Integrate all data sources into a unified, clean dataset
        """
        print("🔄 Integrating all data sources...")
        
        # Clean data from all sources
        fastf1_data = self.clean_fastf1_data()
        openf1_data = self.clean_openf1_data()
        historical_data = self.clean_historical_data()
        
        # Combine all data sources
        all_dataframes = []
        
        if not fastf1_data.empty:
            all_dataframes.append(fastf1_data)
        if not openf1_data.empty:
            all_dataframes.append(openf1_data)
        if not historical_data.empty:
            all_dataframes.append(historical_data)
        
        if not all_dataframes:
            print("❌ No data sources available")
            return pd.DataFrame()
        
        # Merge all data sources
        integrated_data = pd.concat(all_dataframes, ignore_index=True, sort=False)
        
        # Standardize and clean the integrated dataset
        integrated_data = self._standardize_integrated_data(integrated_data)
        
        # Save the integrated dataset
        output_path = os.path.join(self.data_dir, 'integrated_f1_data.csv')
        integrated_data.to_csv(output_path, index=False)
        
        print(f"✅ Data integration complete: {len(integrated_data)} records")
        print(f"💾 Integrated data saved to: {output_path}")
        
        return integrated_data
    
    def _standardize_integrated_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize the integrated dataset
        """
        print("🔧 Standardizing integrated dataset...")
        
        # Standardize column names
        column_mapping = {
            'event_name': 'circuit',
            'location': 'circuit_location',
            'driver_name': 'driver_name',
            'team_name': 'team_name',
            'position': 'finishing_position',
            'position_race': 'finishing_position',
            'position_quali': 'qualifying_position',
            'grid_position': 'grid_position',
            'points': 'points',
            'year': 'year',
            'round': 'round'
        }
        
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns and new_col not in df.columns:
                df[new_col] = df[old_col]
        
        # Handle missing values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            df[col] = df[col].fillna(df[col].median())
        
        # Handle categorical columns
        categorical_columns = df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            df[col] = df[col].fillna('Unknown')
        
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Sort by year and round
        if 'year' in df.columns and 'round' in df.columns:
            df = df.sort_values(['year', 'round', 'finishing_position'])
        
        print(f"✅ Dataset standardized: {len(df)} records, {len(df.columns)} columns")
        
        return df
    
    def validate_data_quality(self, df: pd.DataFrame) -> Dict:
        """
        Validate the quality of the integrated dataset
        """
        print("🔍 Validating data quality...")
        
        validation_report = {
            'total_records': len(df),
            'total_columns': len(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
            'duplicate_records': df.duplicated().sum(),
            'data_types': df.dtypes.to_dict(),
            'year_range': (df['year'].min(), df['year'].max()) if 'year' in df.columns else None,
            'circuit_count': df['circuit'].nunique() if 'circuit' in df.columns else 0,
            'driver_count': df['driver_name'].nunique() if 'driver_name' in df.columns else 0,
            'team_count': df['team_name'].nunique() if 'team_name' in df.columns else 0
        }
        
        # Print validation report
        print("📊 Data Quality Report:")
        print(f"  Total Records: {validation_report['total_records']:,}")
        print(f"  Total Columns: {validation_report['total_columns']}")
        print(f"  Year Range: {validation_report['year_range']}")
        print(f"  Unique Circuits: {validation_report['circuit_count']}")
        print(f"  Unique Drivers: {validation_report['driver_count']}")
        print(f"  Unique Teams: {validation_report['team_count']}")
        print(f"  Duplicate Records: {validation_report['duplicate_records']}")
        
        return validation_report

def main():
    """Main function to run the comprehensive data pipeline"""
    print("🚀 Starting Comprehensive F1 Data Pipeline")
    print("=" * 50)
    
    # Initialize pipeline
    pipeline = ComprehensiveF1DataPipeline()
    
    # Integrate all data sources
    integrated_data = pipeline.integrate_all_data_sources()
    
    if not integrated_data.empty:
        # Validate data quality
        validation_report = pipeline.validate_data_quality(integrated_data)
        
        print("\n✅ Comprehensive F1 Data Pipeline Complete!")
        print("=" * 50)
        print("Next steps:")
        print("1. Run pattern recognition analysis")
        print("2. Train prediction models")
        print("3. Validate prediction accuracy")
    else:
        print("❌ Data pipeline failed - no data available")

if __name__ == "__main__":
    main()
