#!/usr/bin/env python3
"""
Advanced Feature Engineering for F1 Predictions
Adds sophisticated features to improve prediction accuracy
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import time

class AdvancedF1Features:
    def __init__(self):
        self.circuit_characteristics = {
            'Monaco': {'overtaking_difficulty': 5, 'tire_wear': 2, 'fuel_sensitivity': 3, 'track_evolution': 4},
            'Silverstone': {'overtaking_difficulty': 2, 'tire_wear': 4, 'fuel_sensitivity': 3, 'track_evolution': 3},
            'Melbourne': {'overtaking_difficulty': 3, 'tire_wear': 3, 'fuel_sensitivity': 2, 'track_evolution': 3},
            'Spa-Francorchamps': {'overtaking_difficulty': 2, 'tire_wear': 3, 'fuel_sensitivity': 4, 'track_evolution': 2},
            'Monza': {'overtaking_difficulty': 2, 'tire_wear': 2, 'fuel_sensitivity': 4, 'track_evolution': 2},
            # Add more circuits...
        }
    
    def add_driver_momentum_features(self, df, results_history):
        """Add driver momentum and form features"""
        features = []
        
        for _, row in df.iterrows():
            driver_name = row['driver_name']
            session_key = row.get('session_key', '')
            
            # Get driver's recent results
            driver_results = results_history[
                (results_history['driver_name'] == driver_name) & 
                (results_history['session_key'] < session_key)
            ].sort_values('date').tail(10)
            
            if len(driver_results) >= 3:
                # Recent form (last 3 races)
                recent_positions = driver_results['position'].tail(3)
                form_last3 = recent_positions.mean()
                form_trend = np.polyfit(range(len(recent_positions)), recent_positions, 1)[0]  # Slope
                
                # DNF rate
                dnf_rate = (driver_results['position'] > 20).sum() / len(driver_results)
                
                # Points consistency
                points_variance = driver_results['points'].var()
                
                # Qualifying vs Race performance
                quali_race_delta = (driver_results['grid_position'] - driver_results['position']).mean()
                
                # Track-specific form
                circuit = row['circuit']
                track_results = driver_results[driver_results['circuit'] == circuit]
                track_form = track_results['position'].mean() if len(track_results) > 0 else form_last3
                
                features.append({
                    'driver_form_last3': form_last3,
                    'driver_form_trend': form_trend,
                    'driver_dnf_rate': dnf_rate,
                    'driver_points_consistency': 1 / (1 + points_variance),  # Higher = more consistent
                    'driver_quali_race_delta': quali_race_delta,
                    'driver_track_form': track_form
                })
            else:
                # Default values for new drivers
                features.append({
                    'driver_form_last3': 10.0,
                    'driver_form_trend': 0.0,
                    'driver_dnf_rate': 0.1,
                    'driver_points_consistency': 0.5,
                    'driver_quali_race_delta': 0.0,
                    'driver_track_form': 10.0
                })
        
        return pd.DataFrame(features)
    
    def add_team_features(self, df, results_history):
        """Add advanced team performance features"""
        features = []
        
        for _, row in df.iterrows():
            team_name = row['team_name']
            session_key = row.get('session_key', '')
            
            # Get team's recent results
            team_results = results_history[
                (results_history['team_name'] == team_name) & 
                (results_history['session_key'] < session_key)
            ].sort_values('date').tail(10)
            
            if len(team_results) >= 3:
                # Team pace development
                recent_positions = team_results.groupby('session_key')['position'].mean()
                pace_trend = np.polyfit(range(len(recent_positions)), recent_positions, 1)[0]
                
                # Team reliability
                reliability_score = 1 - (team_results['position'] > 20).sum() / len(team_results)
                
                # Team consistency (both cars performing similarly)
                team_consistency = 1 / (1 + team_results.groupby('session_key')['position'].std().mean())
                
                # Championship pressure (teams fighting for specific positions)
                current_points = team_results['team_points'].iloc[-1] if len(team_results) > 0 else 0
                championship_pressure = self.calculate_championship_pressure(current_points)
                
                features.append({
                    'team_pace_trend': pace_trend,
                    'team_reliability': reliability_score,
                    'team_consistency': team_consistency,
                    'team_championship_pressure': championship_pressure
                })
            else:
                features.append({
                    'team_pace_trend': 0.0,
                    'team_reliability': 0.9,
                    'team_consistency': 0.5,
                    'team_championship_pressure': 0.5
                })
        
        return pd.DataFrame(features)
    
    def add_circuit_specific_features(self, df):
        """Add circuit-specific performance features"""
        features = []
        
        for _, row in df.iterrows():
            circuit = row['circuit']
            characteristics = self.circuit_characteristics.get(circuit, {
                'overtaking_difficulty': 3,
                'tire_wear': 3,
                'fuel_sensitivity': 3,
                'track_evolution': 3
            })
            
            # Grid position impact varies by circuit
            grid_position = row['grid_position']
            grid_impact = grid_position * characteristics['overtaking_difficulty'] / 5
            
            features.append({
                'circuit_overtaking_difficulty': characteristics['overtaking_difficulty'],
                'circuit_tire_wear': characteristics['tire_wear'],
                'circuit_fuel_sensitivity': characteristics['fuel_sensitivity'],
                'circuit_track_evolution': characteristics['track_evolution'],
                'grid_position_impact': grid_impact
            })
        
        return pd.DataFrame(features)
    
    def add_weather_impact_features(self, df):
        """Add weather impact features"""
        features = []
        
        for _, row in df.iterrows():
            # Weather impact on different aspects
            rain_impact = self.calculate_rain_impact(row.get('rainfall', 0))
            temp_impact = self.calculate_temperature_impact(row.get('air_temperature', 20))
            wind_impact = self.calculate_wind_impact(row.get('wind_speed', 5))
            
            # Combined weather complexity
            weather_complexity = (rain_impact + temp_impact + wind_impact) / 3
            
            features.append({
                'weather_rain_impact': rain_impact,
                'weather_temp_impact': temp_impact,
                'weather_wind_impact': wind_impact,
                'weather_complexity': weather_complexity
            })
        
        return pd.DataFrame(features)
    
    def add_strategic_features(self, df):
        """Add strategic features"""
        features = []
        
        for _, row in df.iterrows():
            # Tire strategy prediction
            tire_strategy_risk = self.predict_tire_strategy_risk(row)
            
            # Pit stop window optimization
            pit_window_advantage = self.calculate_pit_window_advantage(row)
            
            # Safety car probability
            safety_car_probability = self.estimate_safety_car_probability(row)
            
            features.append({
                'tire_strategy_risk': tire_strategy_risk,
                'pit_window_advantage': pit_window_advantage,
                'safety_car_probability': safety_car_probability
            })
        
        return pd.DataFrame(features)
    
    def calculate_championship_pressure(self, current_points):
        """Calculate championship pressure based on points"""
        # Normalize points to 0-1 scale where higher points = more pressure
        max_points = 800  # Approximate season max
        return min(current_points / max_points, 1.0)
    
    def calculate_rain_impact(self, rainfall):
        """Calculate rain impact on race difficulty"""
        if rainfall < 0.1:
            return 0.0  # Dry
        elif rainfall < 1.0:
            return 0.3  # Light rain
        elif rainfall < 5.0:
            return 0.7  # Moderate rain
        else:
            return 1.0  # Heavy rain
    
    def calculate_temperature_impact(self, temperature):
        """Calculate temperature impact (optimal ~20-25°C)"""
        optimal_temp = 22.5
        return abs(temperature - optimal_temp) / 30.0
    
    def calculate_wind_impact(self, wind_speed):
        """Calculate wind impact on race difficulty"""
        return min(wind_speed / 20.0, 1.0)
    
    def predict_tire_strategy_risk(self, row):
        """Predict tire strategy risk based on conditions"""
        # Simplified tire strategy risk calculation
        base_risk = 0.3
        
        # Weather adds risk
        rainfall = row.get('rainfall', 0)
        weather_risk = self.calculate_rain_impact(rainfall) * 0.4
        
        # Track characteristics
        circuit = row['circuit']
        circuit_risk = self.circuit_characteristics.get(circuit, {}).get('tire_wear', 3) / 5 * 0.3
        
        return min(base_risk + weather_risk + circuit_risk, 1.0)
    
    def calculate_pit_window_advantage(self, row):
        """Calculate pit window advantage based on grid position"""
        grid_pos = row['grid_position']
        # Middle of pack often has more strategic flexibility
        if 6 <= grid_pos <= 12:
            return 0.8
        elif 3 <= grid_pos <= 5 or 13 <= grid_pos <= 16:
            return 0.5
        else:
            return 0.2
    
    def estimate_safety_car_probability(self, row):
        """Estimate safety car probability"""
        circuit = row['circuit']
        base_probability = 0.3
        
        # Some circuits have higher SC probability
        high_sc_circuits = ['Monaco', 'Baku', 'Singapore', 'Jeddah']
        if circuit in high_sc_circuits:
            base_probability = 0.6
        
        # Weather increases probability
        rainfall = row.get('rainfall', 0)
        weather_increase = self.calculate_rain_impact(rainfall) * 0.3
        
        return min(base_probability + weather_increase, 1.0)
    
    def generate_all_features(self, df, results_history):
        """Generate all advanced features"""
        print("🔧 Generating advanced features...")
        
        # Add all feature sets
        driver_features = self.add_driver_momentum_features(df, results_history)
        team_features = self.add_team_features(df, results_history)
        circuit_features = self.add_circuit_specific_features(df)
        weather_features = self.add_weather_impact_features(df)
        strategic_features = self.add_strategic_features(df)
        
        # Combine all features
        enhanced_df = pd.concat([
            df,
            driver_features,
            team_features,
            circuit_features,
            weather_features,
            strategic_features
        ], axis=1)
        
        print(f"✅ Added {len(driver_features.columns) + len(team_features.columns) + len(circuit_features.columns) + len(weather_features.columns) + len(strategic_features.columns)} new features")
        
        return enhanced_df

# Example usage
if __name__ == "__main__":
    # This would be integrated into your main prediction pipeline
    feature_engineer = AdvancedF1Features()
    print("Advanced F1 Feature Engineering Module Ready!")
