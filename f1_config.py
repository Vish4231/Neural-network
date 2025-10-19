#!/usr/bin/env python3
"""
F1 Configuration System
Contains all circuits, drivers, teams, and configuration data
"""

import pandas as pd
from typing import Dict, List, Optional, Any
import json
import os

class F1Config:
    """Main configuration class"""
    def __init__(self):
        self.version = "1.0.0"
        self.default_year = 2025
        self.output_dir = "predictions"
        self.cache_dir = "fastf1_cache"
        self.data_dir = "data"
        
    def get_output_dir(self) -> str:
        """Get output directory, create if needed"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        return self.output_dir

class CircuitDatabase:
    """Database of F1 circuits and their characteristics"""
    
    def __init__(self):
        self.circuits = {
            # 2025 F1 Calendar Circuits
            'bahrain': {
                'name': 'Bahrain International Circuit',
                'country': 'Bahrain',
                'city': 'Sakhir',
                'length_km': 5.412,
                'turns': 15,
                'lap_record': '1:31.447',
                'altitude': 7,
                'downforce_setup': 0.6,
                'drs_zones': 3,
                'overtaking_difficulty': 6,
                'tire_degradation': 'high',
                'safety_car_prob': 0.3,
                'weather_variability': 0.2,
                'grip_level': 7,
                'track_evolution': 'high'
            },
            'saudi': {
                'name': 'Jeddah Corniche Circuit',
                'country': 'Saudi Arabia',
                'city': 'Jeddah',
                'length_km': 6.174,
                'turns': 27,
                'lap_record': '1:30.734',
                'altitude': 15,
                'downforce_setup': 0.7,
                'drs_zones': 3,
                'overtaking_difficulty': 4,
                'tire_degradation': 'medium',
                'safety_car_prob': 0.6,
                'weather_variability': 0.1,
                'grip_level': 8,
                'track_evolution': 'medium'
            },
            'australia': {
                'name': 'Albert Park Circuit',
                'country': 'Australia',
                'city': 'Melbourne',
                'length_km': 5.278,
                'turns': 14,
                'lap_record': '1:24.125',
                'altitude': 2,
                'downforce_setup': 0.65,
                'drs_zones': 3,
                'overtaking_difficulty': 5,
                'tire_degradation': 'medium',
                'safety_car_prob': 0.4,
                'weather_variability': 0.6,
                'grip_level': 8,
                'track_evolution': 'high'
            },
            'japan': {
                'name': 'Suzuka International Racing Course',
                'country': 'Japan',
                'city': 'Suzuka',
                'length_km': 5.807,
                'turns': 18,
                'lap_record': '1:30.983',
                'altitude': 45,
                'downforce_setup': 0.8,
                'drs_zones': 1,
                'overtaking_difficulty': 8,
                'tire_degradation': 'medium',
                'safety_car_prob': 0.3,
                'weather_variability': 0.7,
                'grip_level': 9,
                'track_evolution': 'medium'
            },
            'china': {
                'name': 'Shanghai International Circuit',
                'country': 'China',
                'city': 'Shanghai',
                'length_km': 5.451,
                'turns': 16,
                'lap_record': '1:32.238',
                'altitude': 5,
                'downforce_setup': 0.7,
                'drs_zones': 2,
                'overtaking_difficulty': 6,
                'tire_degradation': 'medium',
                'safety_car_prob': 0.25,
                'weather_variability': 0.5,
                'grip_level': 7,
                'track_evolution': 'medium'
            },
            'miami': {
                'name': 'Miami International Autodrome',
                'country': 'USA',
                'city': 'Miami Gardens',
                'length_km': 5.41,
                'turns': 19,
                'lap_record': '1:31.361',
                'altitude': 3,
                'downforce_setup': 0.6,
                'drs_zones': 3,
                'overtaking_difficulty': 5,
                'tire_degradation': 'high',
                'safety_car_prob': 0.4,
                'weather_variability': 0.6,
                'grip_level': 6,
                'track_evolution': 'high'
            },
            'imola': {
                'name': 'Autodromo Enzo e Dino Ferrari',
                'country': 'Italy',
                'city': 'Imola',
                'length_km': 4.909,
                'turns': 19,
                'lap_record': '1:15.484',
                'altitude': 37,
                'downforce_setup': 0.8,
                'drs_zones': 2,
                'overtaking_difficulty': 9,
                'tire_degradation': 'medium',
                'safety_car_prob': 0.2,
                'weather_variability': 0.4,
                'grip_level': 8,
                'track_evolution': 'low'
            },
            'monaco': {
                'name': 'Circuit de Monaco',
                'country': 'Monaco',
                'city': 'Monte Carlo',
                'length_km': 3.337,
                'turns': 19,
                'lap_record': '1:12.909',
                'altitude': 7,
                'downforce_setup': 1.0,
                'drs_zones': 1,
                'overtaking_difficulty': 10,
                'tire_degradation': 'low',
                'safety_car_prob': 0.8,
                'weather_variability': 0.2,
                'grip_level': 9,
                'track_evolution': 'very_high'
            },
            'canada': {
                'name': 'Circuit Gilles-Villeneuve',
                'country': 'Canada',
                'city': 'Montreal',
                'length_km': 4.361,
                'turns': 14,
                'lap_record': '1:13.078',
                'altitude': 20,
                'downforce_setup': 0.4,
                'drs_zones': 2,
                'overtaking_difficulty': 4,
                'tire_degradation': 'low',
                'safety_car_prob': 0.5,
                'weather_variability': 0.6,
                'grip_level': 6,
                'track_evolution': 'high'
            },
            'spain': {
                'name': 'Circuit de Barcelona-Catalunya',
                'country': 'Spain',
                'city': 'Barcelona',
                'length_km': 4.675,
                'turns': 16,
                'lap_record': '1:18.149',
                'altitude': 109,
                'downforce_setup': 0.75,
                'drs_zones': 2,
                'overtaking_difficulty': 8,
                'tire_degradation': 'high',
                'safety_car_prob': 0.15,
                'weather_variability': 0.3,
                'grip_level': 9,
                'track_evolution': 'medium'
            },
            'austria': {
                'name': 'Red Bull Ring',
                'country': 'Austria',
                'city': 'Spielberg',
                'length_km': 4.318,
                'turns': 10,
                'lap_record': '1:05.619',
                'altitude': 678,
                'downforce_setup': 0.5,
                'drs_zones': 3,
                'overtaking_difficulty': 3,
                'tire_degradation': 'medium',
                'safety_car_prob': 0.3,
                'weather_variability': 0.7,
                'grip_level': 7,
                'track_evolution': 'medium'
            },
            'silverstone': {
                'name': 'Silverstone Circuit',
                'country': 'United Kingdom',
                'city': 'Silverstone',
                'length_km': 5.891,
                'turns': 18,
                'lap_record': '1:27.097',
                'altitude': 153,
                'downforce_setup': 0.6,
                'drs_zones': 2,
                'overtaking_difficulty': 5,
                'tire_degradation': 'high',
                'safety_car_prob': 0.25,
                'weather_variability': 0.8,
                'grip_level': 8,
                'track_evolution': 'medium'
            },
            'hungary': {
                'name': 'Hungaroring',
                'country': 'Hungary',
                'city': 'Budapest',
                'length_km': 4.381,
                'turns': 14,
                'lap_record': '1:16.627',
                'altitude': 264,
                'downforce_setup': 0.9,
                'drs_zones': 2,
                'overtaking_difficulty': 9,
                'tire_degradation': 'medium',
                'safety_car_prob': 0.2,
                'weather_variability': 0.5,
                'grip_level': 7,
                'track_evolution': 'high'
            },
            'spa': {
                'name': 'Circuit de Spa-Francorchamps',
                'country': 'Belgium',
                'city': 'Spa-Francorchamps',
                'length_km': 7.004,
                'turns': 19,
                'lap_record': '1:46.286',
                'altitude': 401,
                'downforce_setup': 0.4,
                'drs_zones': 2,
                'overtaking_difficulty': 3,
                'tire_degradation': 'medium',
                'safety_car_prob': 0.3,
                'weather_variability': 0.9,
                'grip_level': 8,
                'track_evolution': 'low'
            },
            'netherlands': {
                'name': 'Circuit Zandvoort',
                'country': 'Netherlands',
                'city': 'Zandvoort',
                'length_km': 4.259,
                'turns': 14,
                'lap_record': '1:11.097',
                'altitude': 6,
                'downforce_setup': 0.8,
                'drs_zones': 1,
                'overtaking_difficulty': 8,
                'tire_degradation': 'medium',
                'safety_car_prob': 0.2,
                'weather_variability': 0.6,
                'grip_level': 8,
                'track_evolution': 'medium'
            },
            'monza': {
                'name': 'Autodromo Nazionale Monza',
                'country': 'Italy',
                'city': 'Monza',
                'length_km': 5.793,
                'turns': 11,
                'lap_record': '1:21.046',
                'altitude': 162,
                'downforce_setup': 0.3,
                'drs_zones': 2,
                'overtaking_difficulty': 2,
                'tire_degradation': 'medium',
                'safety_car_prob': 0.25,
                'weather_variability': 0.4,
                'grip_level': 8,
                'track_evolution': 'low'
            },
            'azerbaijan': {
                'name': 'Baku City Circuit',
                'country': 'Azerbaijan',
                'city': 'Baku',
                'length_km': 6.003,
                'turns': 20,
                'lap_record': '1:43.009',
                'altitude': -1,
                'downforce_setup': 0.5,
                'drs_zones': 2,
                'overtaking_difficulty': 4,
                'tire_degradation': 'medium',
                'safety_car_prob': 0.6,
                'weather_variability': 0.3,
                'grip_level': 6,
                'track_evolution': 'high'
            },
            'singapore': {
                'name': 'Marina Bay Street Circuit',
                'country': 'Singapore',
                'city': 'Singapore',
                'length_km': 5.063,
                'turns': 23,
                'lap_record': '1:35.867',
                'altitude': 18,
                'downforce_setup': 0.8,
                'drs_zones': 3,
                'overtaking_difficulty': 7,
                'tire_degradation': 'medium',
                'safety_car_prob': 0.7,
                'weather_variability': 0.5,
                'grip_level': 7,
                'track_evolution': 'very_high'
            },
            'cota': {
                'name': 'Circuit of the Americas',
                'country': 'USA',
                'city': 'Austin',
                'length_km': 5.513,
                'turns': 20,
                'lap_record': '1:36.169',
                'altitude': 161,
                'downforce_setup': 0.7,
                'drs_zones': 2,
                'overtaking_difficulty': 6,
                'tire_degradation': 'medium',
                'safety_car_prob': 0.4,
                'weather_variability': 0.5,
                'grip_level': 8,
                'track_evolution': 'medium'
            },
            'mexico': {
                'name': 'Autodromo Hermanos Rodriguez',
                'country': 'Mexico',
                'city': 'Mexico City',
                'length_km': 4.304,
                'turns': 17,
                'lap_record': '1:17.774',
                'altitude': 2285,
                'downforce_setup': 0.6,
                'drs_zones': 3,
                'overtaking_difficulty': 4,
                'tire_degradation': 'low',
                'safety_car_prob': 0.3,
                'weather_variability': 0.4,
                'grip_level': 6,
                'track_evolution': 'medium'
            },
            'brazil': {
                'name': 'Autodromo Jose Carlos Pace',
                'country': 'Brazil',
                'city': 'Sao Paulo',
                'length_km': 4.309,
                'turns': 15,
                'lap_record': '1:10.540',
                'altitude': 760,
                'downforce_setup': 0.6,
                'drs_zones': 2,
                'overtaking_difficulty': 5,
                'tire_degradation': 'medium',
                'safety_car_prob': 0.4,
                'weather_variability': 0.8,
                'grip_level': 7,
                'track_evolution': 'high'
            },
            'las_vegas': {
                'name': 'Las Vegas Street Circuit',
                'country': 'USA',
                'city': 'Las Vegas',
                'length_km': 6.201,
                'turns': 17,
                'lap_record': '1:35.490',
                'altitude': 610,
                'downforce_setup': 0.4,
                'drs_zones': 3,
                'overtaking_difficulty': 4,
                'tire_degradation': 'medium',
                'safety_car_prob': 0.3,
                'weather_variability': 0.2,
                'grip_level': 6,
                'track_evolution': 'high'
            },
            'qatar': {
                'name': 'Lusail International Circuit',
                'country': 'Qatar',
                'city': 'Lusail',
                'length_km': 5.419,
                'turns': 16,
                'lap_record': '1:24.319',
                'altitude': 0,
                'downforce_setup': 0.6,
                'drs_zones': 2,
                'overtaking_difficulty': 5,
                'tire_degradation': 'high',
                'safety_car_prob': 0.2,
                'weather_variability': 0.1,
                'grip_level': 7,
                'track_evolution': 'medium'
            },
            'abu_dhabi': {
                'name': 'Yas Marina Circuit',
                'country': 'UAE',
                'city': 'Abu Dhabi',
                'length_km': 5.281,
                'turns': 16,
                'lap_record': '1:26.103',
                'altitude': 3,
                'downforce_setup': 0.7,
                'drs_zones': 2,
                'overtaking_difficulty': 7,
                'tire_degradation': 'medium',
                'safety_car_prob': 0.25,
                'weather_variability': 0.1,
                'grip_level': 8,
                'track_evolution': 'low'
            }
        }
    
    def get_all_circuits(self) -> Dict[str, Dict]:
        """Get all circuits"""
        return self.circuits
    
    def get_circuit_data(self, circuit_key: str) -> Dict:
        """Get specific circuit data"""
        return self.circuits.get(circuit_key, {})
    
    def find_circuit(self, search_term: str) -> Optional[str]:
        """Find circuit by name or key (case insensitive)"""
        search_term = search_term.lower().strip()
        
        # Direct key match
        if search_term in self.circuits:
            return search_term
        
        # Search in circuit names and aliases
        aliases = {
            'spa-francorchamps': 'spa',
            'spa francorchamps': 'spa',
            'belgian gp': 'spa',
            'belgium': 'spa',
            'monaco gp': 'monaco',
            'monte carlo': 'monaco',
            'silverstone': 'silverstone',
            'british gp': 'silverstone',
            'britain': 'silverstone',
            'uk': 'silverstone',
            'monza': 'monza',
            'italian gp': 'monza',
            'italy': 'monza',
            'suzuka': 'japan',
            'japanese gp': 'japan',
            'interlagos': 'brazil',
            'brazilian gp': 'brazil',
            'sao paulo': 'brazil',
            'red bull ring': 'austria',
            'austrian gp': 'austria',
            'hungaroring': 'hungary',
            'hungarian gp': 'hungary',
            'barcelona': 'spain',
            'spanish gp': 'spain',
            'catalunya': 'spain',
            'zandvoort': 'netherlands',
            'dutch gp': 'netherlands',
            'albert park': 'australia',
            'australian gp': 'australia',
            'melbourne': 'australia',
            'circuit gilles villeneuve': 'canada',
            'canadian gp': 'canada',
            'montreal': 'canada',
            'paul ricard': 'france',
            'french gp': 'france',
            'baku': 'azerbaijan',
            'azerbaijan gp': 'azerbaijan',
            'marina bay': 'singapore',
            'singapore gp': 'singapore',
            'cota': 'cota',
            'austin': 'cota',
            'circuit of the americas': 'cota',
            'us gp': 'cota',
            'usa': 'cota',
            'hermanos rodriguez': 'mexico',
            'mexican gp': 'mexico',
            'mexico city': 'mexico',
            'las vegas': 'las_vegas',
            'vegas': 'las_vegas',
            'yas marina': 'abu_dhabi',
            'abu dhabi gp': 'abu_dhabi',
            'uae': 'abu_dhabi',
            'bahrain gp': 'bahrain',
            'sakhir': 'bahrain',
            'jeddah': 'saudi',
            'saudi arabia': 'saudi',
            'saudi arabian gp': 'saudi',
            'miami gp': 'miami',
            'miami gardens': 'miami',
            'imola gp': 'imola',
            'emilia romagna': 'imola',
            'lusail': 'qatar',
            'qatar gp': 'qatar',
            'shanghai': 'china',
            'chinese gp': 'china'
        }
        
        if search_term in aliases:
            return aliases[search_term]
        
        # Partial name matching
        for key, data in self.circuits.items():
            if search_term in data['name'].lower() or search_term in data.get('city', '').lower():
                return key
        
        return None
    
    def get_circuit_list_for_display(self) -> List[Dict]:
        """Get circuits formatted for display"""
        circuits = []
        for key, data in self.circuits.items():
            circuits.append({
                'key': key,
                'name': data['name'],
                'country': data['country'],
                'length_km': data['length_km'],
                'turns': data['turns'],
                'difficulty': data['overtaking_difficulty']
            })
        return sorted(circuits, key=lambda x: x['name'])

class DriverDatabase:
    """Database of F1 drivers and teams"""
    
    def __init__(self):
        # 2025 Official F1 Driver Lineup (confirmed changes)
        self.drivers_2025 = [
            # Red Bull Racing (Verstappen stays, Tsunoda promoted)
            {'driver_name': 'Max Verstappen', 'team_name': 'Red Bull Racing', 'driver_number': 1},
            {'driver_name': 'Yuki Tsunoda', 'team_name': 'Red Bull Racing', 'driver_number': 22},
            
            # Ferrari (Hamilton joins from Mercedes!)
            {'driver_name': 'Charles Leclerc', 'team_name': 'Ferrari', 'driver_number': 16},
            {'driver_name': 'Lewis Hamilton', 'team_name': 'Ferrari', 'driver_number': 44},
            
            # Mercedes (Russell leads, Antonelli debuts)
            {'driver_name': 'George Russell', 'team_name': 'Mercedes', 'driver_number': 63},
            {'driver_name': 'Andrea Kimi Antonelli', 'team_name': 'Mercedes', 'driver_number': 12},
            
            # McLaren (unchanged)
            {'driver_name': 'Lando Norris', 'team_name': 'McLaren', 'driver_number': 4},
            {'driver_name': 'Oscar Piastri', 'team_name': 'McLaren', 'driver_number': 81},
            
            # Aston Martin (unchanged)
            {'driver_name': 'Fernando Alonso', 'team_name': 'Aston Martin', 'driver_number': 14},
            {'driver_name': 'Lance Stroll', 'team_name': 'Aston Martin', 'driver_number': 18},
            
            # Alpine (Gasly stays, Colapinto joins from Williams)
            {'driver_name': 'Pierre Gasly', 'team_name': 'Alpine', 'driver_number': 10},
            {'driver_name': 'Franco Colapinto', 'team_name': 'Alpine', 'driver_number': 43},
            
            # Williams (Albon stays, Sainz joins from Ferrari!)
            {'driver_name': 'Alexander Albon', 'team_name': 'Williams', 'driver_number': 23},
            {'driver_name': 'Carlos Sainz', 'team_name': 'Williams', 'driver_number': 55},
            
            # Racing Bulls (formerly RB/AlphaTauri - Lawson and Hadjar)
            {'driver_name': 'Liam Lawson', 'team_name': 'Racing Bulls', 'driver_number': 30},
            {'driver_name': 'Isack Hadjar', 'team_name': 'Racing Bulls', 'driver_number': 6},
            
            # Haas (Ocon from Alpine, Bearman from F2)
            {'driver_name': 'Esteban Ocon', 'team_name': 'Haas', 'driver_number': 31},
            {'driver_name': 'Oliver Bearman', 'team_name': 'Haas', 'driver_number': 87},
            
            # Sauber (Hulkenberg stays, Bortoleto from F2)
            {'driver_name': 'Nico Hulkenberg', 'team_name': 'Sauber', 'driver_number': 27},
            {'driver_name': 'Gabriel Bortoleto', 'team_name': 'Sauber', 'driver_number': 24}
        ]
        
        # Driver performance ratings (0-1 scale)
        self.driver_ratings = {
            'Max Verstappen': {'skill': 0.98, 'consistency': 0.96, 'racecraft': 0.97},
            'Lewis Hamilton': {'skill': 0.95, 'consistency': 0.92, 'racecraft': 0.98},
            'Charles Leclerc': {'skill': 0.93, 'consistency': 0.88, 'racecraft': 0.91},
            'Lando Norris': {'skill': 0.91, 'consistency': 0.90, 'racecraft': 0.89},
            'George Russell': {'skill': 0.89, 'consistency': 0.91, 'racecraft': 0.87},
            'Carlos Sainz': {'skill': 0.87, 'consistency': 0.89, 'racecraft': 0.88},
            'Fernando Alonso': {'skill': 0.92, 'consistency': 0.85, 'racecraft': 0.96},
            'Oscar Piastri': {'skill': 0.86, 'consistency': 0.87, 'racecraft': 0.85},
            'Pierre Gasly': {'skill': 0.84, 'consistency': 0.82, 'racecraft': 0.83},
            'Alexander Albon': {'skill': 0.83, 'consistency': 0.86, 'racecraft': 0.82},
            'Esteban Ocon': {'skill': 0.81, 'consistency': 0.84, 'racecraft': 0.80},
            'Lance Stroll': {'skill': 0.76, 'consistency': 0.78, 'racecraft': 0.75},
            'Nico Hulkenberg': {'skill': 0.82, 'consistency': 0.85, 'racecraft': 0.81},
            'Yuki Tsunoda': {'skill': 0.80, 'consistency': 0.75, 'racecraft': 0.78},
            'Liam Lawson': {'skill': 0.78, 'consistency': 0.76, 'racecraft': 0.77},
            'Andrea Kimi Antonelli': {'skill': 0.82, 'consistency': 0.75, 'racecraft': 0.79},
            'Oliver Bearman': {'skill': 0.79, 'consistency': 0.77, 'racecraft': 0.76},
            'Franco Colapinto': {'skill': 0.77, 'consistency': 0.74, 'racecraft': 0.75},
            'Isack Hadjar': {'skill': 0.75, 'consistency': 0.73, 'racecraft': 0.74},
            'Gabriel Bortoleto': {'skill': 0.74, 'consistency': 0.72, 'racecraft': 0.73}
        }
        
        # Team performance ratings (2025 estimates)
        self.team_ratings = {
            'Red Bull Racing': {'car_performance': 0.95, 'strategy': 0.94, 'pit_stops': 0.92, 'development': 0.90},
            'Ferrari': {'car_performance': 0.92, 'strategy': 0.85, 'pit_stops': 0.88, 'development': 0.93},
            'Mercedes': {'car_performance': 0.90, 'strategy': 0.91, 'pit_stops': 0.94, 'development': 0.88},
            'McLaren': {'car_performance': 0.89, 'strategy': 0.87, 'pit_stops': 0.90, 'development': 0.92},
            'Aston Martin': {'car_performance': 0.82, 'strategy': 0.83, 'pit_stops': 0.85, 'development': 0.85},
            'Alpine': {'car_performance': 0.78, 'strategy': 0.80, 'pit_stops': 0.82, 'development': 0.83},
            'Williams': {'car_performance': 0.75, 'strategy': 0.78, 'pit_stops': 0.79, 'development': 0.86},
            'Racing Bulls': {'car_performance': 0.73, 'strategy': 0.76, 'pit_stops': 0.81, 'development': 0.80},
            'Haas': {'car_performance': 0.70, 'strategy': 0.72, 'pit_stops': 0.75, 'development': 0.78},
            'Sauber': {'car_performance': 0.68, 'strategy': 0.70, 'pit_stops': 0.73, 'development': 0.75}
        }
    
    def get_2025_lineup(self) -> pd.DataFrame:
        """Get 2025 F1 driver lineup"""
        return pd.DataFrame(self.drivers_2025)
    
    def get_historical_lineup(self, year: int) -> Optional[pd.DataFrame]:
        """Get historical lineup for a given year (placeholder)"""
        if year == 2024:
            # Return 2024 lineup (simplified)
            drivers_2024 = [
                {'driver_name': 'Max Verstappen', 'team_name': 'Red Bull Racing'},
                {'driver_name': 'Sergio Perez', 'team_name': 'Red Bull Racing'},
                {'driver_name': 'Lewis Hamilton', 'team_name': 'Mercedes'},
                {'driver_name': 'George Russell', 'team_name': 'Mercedes'},
                # ... etc
            ]
            return pd.DataFrame(drivers_2024)
        return None
    
    def get_driver_rating(self, driver_name: str) -> Dict[str, float]:
        """Get driver performance rating"""
        return self.driver_ratings.get(driver_name, {'skill': 0.5, 'consistency': 0.5, 'racecraft': 0.5})
    
    def get_team_rating(self, team_name: str) -> Dict[str, float]:
        """Get team performance rating"""
        return self.team_ratings.get(team_name, {'car_performance': 0.5, 'strategy': 0.5, 'pit_stops': 0.5, 'development': 0.5})
    
    def get_all_drivers(self) -> List[str]:
        """Get list of all 2025 driver names"""
        return [driver['driver_name'] for driver in self.drivers_2025]
    
    def get_all_teams(self) -> List[str]:
        """Get list of all 2025 team names"""
        return list(set(driver['team_name'] for driver in self.drivers_2025))