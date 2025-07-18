#!/usr/bin/env python3
"""
Enhanced OpenF1 API Data Extraction for F1 Predictions
Extracts advanced features from OpenF1 API to improve prediction accuracy
"""

import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import json
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

class EnhancedOpenF1Extractor:
    
    def __init__(self, base_url="https://api.openf1.org/v1"):
        self.base_url = base_url
        self.session_cache = {}
        self.driver_cache = {}
        
    def safe_api_call(self, endpoint: str, params: dict = None, max_retries: int = 3) -> Optional[List]:
        """Make safe API call with retry logic"""
        for attempt in range(max_retries):
            try:
                url = f"{self.base_url}/{endpoint}"
                response = requests.get(url, params=params, timeout=30)
                
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:
                    wait_time = 2 ** attempt
                    print(f"Rate limited, waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"API error {response.status_code}: {response.text[:100]}")
                    return None
                    
            except Exception as e:
                print(f"Request failed (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)
                    
        return None
    
    def get_practice_session_pace(self, session_key: str) -> Dict:
        """Extract pace data from practice sessions"""
        print(f"📊 Extracting practice pace data for session {session_key}")
        
        # Get lap times
        laps = self.safe_api_call("laps", {"session_key": session_key})
        if not laps:
            return {}
            
        df_laps = pd.DataFrame(laps)
        
        # Get drivers info
        drivers = self.safe_api_call("drivers", {"session_key": session_key})
        if not drivers:
            return {}
            
        driver_map = {d['driver_number']: d for d in drivers}
        
        pace_features = {}
        
        for driver_num, driver_info in driver_map.items():
            driver_laps = df_laps[df_laps['driver_number'] == driver_num]
            
            if len(driver_laps) > 0:
                # Clean lap times (remove outliers)
                lap_times = pd.to_numeric(driver_laps['lap_duration'], errors='coerce')
                lap_times = lap_times.dropna()
                
                if len(lap_times) > 3:
                    # Remove top/bottom 10% as outliers
                    q10, q90 = lap_times.quantile([0.1, 0.9])
                    clean_times = lap_times[(lap_times >= q10) & (lap_times <= q90)]
                    
                    if len(clean_times) > 0:
                        pace_features[driver_num] = {
                            'avg_lap_time': clean_times.mean(),
                            'best_lap_time': clean_times.min(),
                            'lap_time_consistency': clean_times.std(),
                            'total_laps': len(driver_laps),
                            'pace_rank': 0  # Will be calculated later
                        }
        
        # Calculate pace rankings
        sorted_drivers = sorted(pace_features.items(), key=lambda x: x[1]['avg_lap_time'])
        for rank, (driver_num, data) in enumerate(sorted_drivers, 1):
            pace_features[driver_num]['pace_rank'] = rank
            
        return pace_features
    
    def get_qualifying_sector_analysis(self, session_key: str) -> Dict:
        """Extract detailed qualifying sector analysis"""
        print(f"🏁 Extracting qualifying sector analysis for session {session_key}")
        
        # Get qualifying laps
        laps = self.safe_api_call("laps", {"session_key": session_key})
        if not laps:
            return {}
            
        df_laps = pd.DataFrame(laps)
        
        # Get drivers info
        drivers = self.safe_api_call("drivers", {"session_key": session_key})
        if not drivers:
            return {}
            
        driver_map = {d['driver_number']: d for d in drivers}
        
        sector_features = {}
        
        for driver_num, driver_info in driver_map.items():
            driver_laps = df_laps[df_laps['driver_number'] == driver_num]
            
            if len(driver_laps) > 0:
                # Get best qualifying lap
                valid_laps = driver_laps.dropna(subset=['lap_duration'])
                
                if len(valid_laps) > 0:
                    best_lap = valid_laps.loc[valid_laps['lap_duration'].idxmin()]
                    
                    sector_features[driver_num] = {
                        'best_lap_time': best_lap['lap_duration'],
                        'sector_1': best_lap.get('duration_sector_1', None),
                        'sector_2': best_lap.get('duration_sector_2', None),
                        'sector_3': best_lap.get('duration_sector_3', None),
                        'is_personal_best': best_lap.get('is_personal_best', False),
                        'segments_sector_1': best_lap.get('segments_sector_1', []),
                        'segments_sector_2': best_lap.get('segments_sector_2', []),
                        'segments_sector_3': best_lap.get('segments_sector_3', [])
                    }
        
        return sector_features
    
    def get_car_telemetry_insights(self, session_key: str, sample_size: int = 1000) -> Dict:
        """Extract car telemetry insights"""
        print(f"🚗 Extracting car telemetry insights for session {session_key}")
        
        # Get car data sample
        car_data = self.safe_api_call("car_data", {
            "session_key": session_key,
            "limit": sample_size
        })
        
        if not car_data:
            return {}
            
        df_car = pd.DataFrame(car_data)
        
        telemetry_features = {}
        
        # Group by driver
        for driver_num in df_car['driver_number'].unique():
            driver_data = df_car[df_car['driver_number'] == driver_num]
            
            if len(driver_data) > 10:  # Need sufficient data points
                telemetry_features[driver_num] = {
                    'avg_speed': driver_data['speed'].mean(),
                    'max_speed': driver_data['speed'].max(),
                    'speed_variance': driver_data['speed'].var(),
                    'avg_throttle': driver_data['throttle'].mean(),
                    'throttle_efficiency': driver_data['throttle'].std(),  # Lower std = smoother
                    'avg_brake': driver_data['brake'].mean(),
                    'brake_efficiency': driver_data['brake'].std(),
                    'avg_gear': driver_data['n_gear'].mean(),
                    'gear_changes': len(driver_data['n_gear'].diff().dropna().nonzero()[0]),
                    'rpm_avg': driver_data['rpm'].mean(),
                    'rpm_variance': driver_data['rpm'].var()
                }
        
        return telemetry_features
    
    def get_pit_stop_strategy_analysis(self, session_key: str) -> Dict:
        """Analyze pit stop strategies"""
        print(f"🏁 Extracting pit stop strategy analysis for session {session_key}")
        
        # Get pit stop data
        pit_data = self.safe_api_call("pit", {"session_key": session_key})
        if not pit_data:
            return {}
            
        df_pit = pd.DataFrame(pit_data)
        
        # Get race control messages for strategy context
        race_control = self.safe_api_call("race_control", {"session_key": session_key})
        
        strategy_features = {}
        
        # Analyze pit stops per driver
        for driver_num in df_pit['driver_number'].unique():
            driver_pits = df_pit[df_pit['driver_number'] == driver_num]
            
            if len(driver_pits) > 0:
                strategy_features[driver_num] = {
                    'pit_count': len(driver_pits),
                    'avg_pit_duration': driver_pits['pit_duration'].mean(),
                    'pit_time_variance': driver_pits['pit_duration'].var(),
                    'first_pit_lap': driver_pits['lap_number'].min(),
                    'last_pit_lap': driver_pits['lap_number'].max(),
                    'pit_timing_strategy': self.analyze_pit_timing(driver_pits, race_control)
                }
        
        return strategy_features
    
    def analyze_pit_timing(self, driver_pits: pd.DataFrame, race_control: List) -> str:
        """Analyze pit timing strategy"""
        if len(driver_pits) == 0:
            return "no_stops"
        
        # Check for safety car periods
        safety_car_periods = []
        if race_control:
            for msg in race_control:
                if 'safety car' in msg.get('message', '').lower():
                    safety_car_periods.append(msg.get('lap_number', 0))
        
        # Analyze timing
        pit_laps = driver_pits['lap_number'].tolist()
        
        # Check if pitted during safety car
        if any(lap in safety_car_periods for lap in pit_laps):
            return "safety_car_strategy"
        
        # Check if early/late strategy
        if len(pit_laps) > 0:
            first_pit = pit_laps[0]
            if first_pit < 15:
                return "early_strategy"
            elif first_pit > 35:
                return "late_strategy"
            else:
                return "standard_strategy"
        
        return "unknown_strategy"
    
    def get_weather_impact_analysis(self, meeting_key: str, session_key: str) -> Dict:
        """Analyze weather impact on performance"""
        print(f"🌤️ Extracting weather impact analysis")
        
        # Get weather data
        weather_data = self.safe_api_call("weather", {
            "meeting_key": meeting_key,
            "session_key": session_key
        })
        
        if not weather_data:
            return {}
            
        df_weather = pd.DataFrame(weather_data)
        
        # Calculate weather trends and impact
        weather_features = {
            'avg_air_temp': df_weather['air_temperature'].mean(),
            'temp_variance': df_weather['air_temperature'].var(),
            'avg_humidity': df_weather['humidity'].mean(),
            'humidity_variance': df_weather['humidity'].var(),
            'avg_rainfall': df_weather['rainfall'].mean(),
            'rain_probability': (df_weather['rainfall'] > 0).mean(),
            'avg_track_temp': df_weather['track_temperature'].mean(),
            'track_temp_variance': df_weather['track_temperature'].var(),
            'avg_wind_speed': df_weather['wind_speed'].mean(),
            'wind_direction_changes': len(df_weather['wind_direction'].diff().dropna().nonzero()[0]),
            'weather_stability': self.calculate_weather_stability(df_weather)
        }
        
        return weather_features
    
    def calculate_weather_stability(self, df_weather: pd.DataFrame) -> float:
        """Calculate weather stability score (0-1, higher = more stable)"""
        stability_factors = []
        
        # Temperature stability
        temp_stability = 1 / (1 + df_weather['air_temperature'].std())
        stability_factors.append(temp_stability)
        
        # Humidity stability
        humidity_stability = 1 / (1 + df_weather['humidity'].std())
        stability_factors.append(humidity_stability)
        
        # Rain consistency
        rain_consistency = 1 - (df_weather['rainfall'] > 0).std()
        stability_factors.append(rain_consistency)
        
        return np.mean(stability_factors)
    
    def get_driver_race_pace_analysis(self, session_key: str) -> Dict:
        """Analyze driver race pace patterns"""
        print(f"🏎️ Extracting driver race pace analysis")
        
        # Get race laps
        laps = self.safe_api_call("laps", {"session_key": session_key})
        if not laps:
            return {}
            
        df_laps = pd.DataFrame(laps)
        
        # Get intervals for position tracking
        intervals = self.safe_api_call("intervals", {"session_key": session_key})
        
        pace_analysis = {}
        
        for driver_num in df_laps['driver_number'].unique():
            driver_laps = df_laps[df_laps['driver_number'] == driver_num].copy()
            
            if len(driver_laps) > 10:  # Need sufficient laps
                # Clean lap times
                driver_laps['lap_duration'] = pd.to_numeric(driver_laps['lap_duration'], errors='coerce')
                driver_laps = driver_laps.dropna(subset=['lap_duration'])
                
                if len(driver_laps) > 5:
                    # Calculate pace metrics
                    pace_analysis[driver_num] = {
                        'avg_race_pace': driver_laps['lap_duration'].mean(),
                        'pace_consistency': driver_laps['lap_duration'].std(),
                        'fastest_lap': driver_laps['lap_duration'].min(),
                        'slowest_lap': driver_laps['lap_duration'].max(),
                        'pace_degradation': self.calculate_pace_degradation(driver_laps),
                        'stint_analysis': self.analyze_stint_performance(driver_laps),
                        'position_changes': self.calculate_position_changes(driver_num, intervals)
                    }
        
        return pace_analysis
    
    def calculate_pace_degradation(self, driver_laps: pd.DataFrame) -> float:
        """Calculate pace degradation over race distance"""
        if len(driver_laps) < 10:
            return 0.0
            
        # Split race into thirds
        total_laps = len(driver_laps)
        third = total_laps // 3
        
        first_third = driver_laps.iloc[:third]['lap_duration'].mean()
        last_third = driver_laps.iloc[-third:]['lap_duration'].mean()
        
        # Return degradation rate (positive = getting slower)
        return (last_third - first_third) / first_third
    
    def analyze_stint_performance(self, driver_laps: pd.DataFrame) -> Dict:
        """Analyze performance by stint"""
        # Simple stint detection (would need pit data for accuracy)
        stint_analysis = {
            'stint_count': 1,  # Placeholder
            'avg_stint_length': len(driver_laps),
            'stint_pace_variation': driver_laps['lap_duration'].std()
        }
        
        return stint_analysis
    
    def calculate_position_changes(self, driver_num: int, intervals: List) -> Dict:
        """Calculate position changes throughout race"""
        if not intervals:
            return {'net_change': 0, 'positions_gained': 0, 'positions_lost': 0}
        
        df_intervals = pd.DataFrame(intervals)
        driver_intervals = df_intervals[df_intervals['driver_number'] == driver_num]
        
        if len(driver_intervals) < 2:
            return {'net_change': 0, 'positions_gained': 0, 'positions_lost': 0}
        
        positions = driver_intervals['position'].tolist()
        
        # Calculate changes
        position_changes = np.diff(positions)
        positions_gained = sum(1 for change in position_changes if change < 0)  # Negative = gained
        positions_lost = sum(1 for change in position_changes if change > 0)    # Positive = lost
        net_change = positions[0] - positions[-1]  # Start - End
        
        return {
            'net_change': net_change,
            'positions_gained': positions_gained,
            'positions_lost': positions_lost
        }
    
    def extract_comprehensive_features(self, session_key: str, meeting_key: str) -> Dict:
        """Extract all advanced features for a session"""
        print(f"🔧 Extracting comprehensive features for session {session_key}")
        
        all_features = {}
        
        # Get basic session info
        sessions = self.safe_api_call("sessions", {"session_key": session_key})
        if sessions:
            session_info = sessions[0]
            session_type = session_info.get('session_type', 'Unknown')
            
            # Extract different features based on session type
            if session_type in ['Practice 1', 'Practice 2', 'Practice 3']:
                all_features['practice_pace'] = self.get_practice_session_pace(session_key)
                all_features['telemetry'] = self.get_car_telemetry_insights(session_key)
                
            elif session_type == 'Qualifying':
                all_features['qualifying_sectors'] = self.get_qualifying_sector_analysis(session_key)
                all_features['telemetry'] = self.get_car_telemetry_insights(session_key)
                
            elif session_type == 'Race':
                all_features['race_pace'] = self.get_driver_race_pace_analysis(session_key)
                all_features['pit_strategy'] = self.get_pit_stop_strategy_analysis(session_key)
                all_features['telemetry'] = self.get_car_telemetry_insights(session_key)
            
            # Weather analysis for all sessions
            all_features['weather'] = self.get_weather_impact_analysis(meeting_key, session_key)
        
        return all_features
    
    def create_feature_dataframe(self, features_dict: Dict, drivers_info: List) -> pd.DataFrame:
        """Convert extracted features to DataFrame format"""
        print("📊 Converting features to DataFrame...")
        
        rows = []
        
        for driver_info in drivers_info:
            driver_num = driver_info['driver_number']
            
            # Base driver info
            row = {
                'driver_number': driver_num,
                'driver_name': driver_info.get('full_name', ''),
                'team_name': driver_info.get('team_name', ''),
                'country_code': driver_info.get('country_code', '')
            }
            
            # Add practice pace features
            if 'practice_pace' in features_dict and driver_num in features_dict['practice_pace']:
                pace_data = features_dict['practice_pace'][driver_num]
                row.update({
                    'practice_avg_lap': pace_data.get('avg_lap_time', 0),
                    'practice_best_lap': pace_data.get('best_lap_time', 0),
                    'practice_consistency': pace_data.get('lap_time_consistency', 0),
                    'practice_pace_rank': pace_data.get('pace_rank', 20)
                })
            
            # Add qualifying features
            if 'qualifying_sectors' in features_dict and driver_num in features_dict['qualifying_sectors']:
                quali_data = features_dict['qualifying_sectors'][driver_num]
                row.update({
                    'quali_best_lap': quali_data.get('best_lap_time', 0),
                    'quali_sector_1': quali_data.get('sector_1', 0),
                    'quali_sector_2': quali_data.get('sector_2', 0),
                    'quali_sector_3': quali_data.get('sector_3', 0)
                })
            
            # Add telemetry features
            if 'telemetry' in features_dict and driver_num in features_dict['telemetry']:
                telemetry_data = features_dict['telemetry'][driver_num]
                row.update({
                    'avg_speed': telemetry_data.get('avg_speed', 0),
                    'max_speed': telemetry_data.get('max_speed', 0),
                    'throttle_efficiency': telemetry_data.get('throttle_efficiency', 0),
                    'brake_efficiency': telemetry_data.get('brake_efficiency', 0),
                    'gear_changes': telemetry_data.get('gear_changes', 0)
                })
            
            # Add race pace features
            if 'race_pace' in features_dict and driver_num in features_dict['race_pace']:
                race_data = features_dict['race_pace'][driver_num]
                row.update({
                    'race_avg_pace': race_data.get('avg_race_pace', 0),
                    'race_consistency': race_data.get('pace_consistency', 0),
                    'pace_degradation': race_data.get('pace_degradation', 0),
                    'net_position_change': race_data.get('position_changes', {}).get('net_change', 0)
                })
            
            # Add pit strategy features
            if 'pit_strategy' in features_dict and driver_num in features_dict['pit_strategy']:
                pit_data = features_dict['pit_strategy'][driver_num]
                row.update({
                    'pit_count': pit_data.get('pit_count', 0),
                    'avg_pit_duration': pit_data.get('avg_pit_duration', 0),
                    'pit_strategy_type': pit_data.get('pit_timing_strategy', 'unknown')
                })
            
            # Add weather features (same for all drivers)
            if 'weather' in features_dict:
                weather_data = features_dict['weather']
                row.update({
                    'weather_stability': weather_data.get('weather_stability', 0.5),
                    'rain_probability': weather_data.get('rain_probability', 0),
                    'temp_variance': weather_data.get('temp_variance', 0)
                })
            
            rows.append(row)
        
        return pd.DataFrame(rows)

def main():
    """Example usage of enhanced OpenF1 extraction"""
    extractor = EnhancedOpenF1Extractor()
    
    # Example: Get enhanced features for a specific session
    session_key = "9158"  # Example session key
    meeting_key = "1217"  # Example meeting key
    
    print("🚀 Starting enhanced OpenF1 feature extraction...")
    
    # Extract comprehensive features
    features = extractor.extract_comprehensive_features(session_key, meeting_key)
    
    # Get drivers info
    drivers = extractor.safe_api_call("drivers", {"session_key": session_key})
    
    if drivers:
        # Create feature DataFrame
        df_features = extractor.create_feature_dataframe(features, drivers)
        
        print(f"✅ Extracted {len(df_features.columns)} features for {len(df_features)} drivers")
        print("\nFeature columns:")
        print(df_features.columns.tolist())
        
        # Save to CSV
        df_features.to_csv("enhanced_openf1_features.csv", index=False)
        print("💾 Features saved to enhanced_openf1_features.csv")
    else:
        print("❌ Could not fetch drivers information")

if __name__ == "__main__":
    main()
