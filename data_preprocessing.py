#!/usr/bin/env python3
"""
F1 Data Preprocessing and Cleaning
Cleans and preprocesses raw FastF1 data for machine learning
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import warnings
from typing import Dict, List, Tuple, Optional
import os

warnings.filterwarnings('ignore')

class F1DataPreprocessor:
    def __init__(self):
        """Initialize data preprocessor"""
        self.scalers = {}
        self.encoders = {}
        self.feature_columns = []
        
        print("🧹 F1 Data Preprocessor Initialized")
    
    def clean_race_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean raw race data"""
        print("🔄 Starting data cleaning...")
        
        cleaned_df = df.copy()
        original_rows = len(cleaned_df)
        
        # 1. Remove clearly invalid entries
        cleaned_df = cleaned_df.dropna(subset=['FullName', 'TeamName', 'Position'])
        
        # 2. Handle DNF/DSQ positions
        cleaned_df['FinishPosition'] = pd.to_numeric(cleaned_df['Position'], errors='coerce')
        
        # 3. Create binary flags for race completion
        cleaned_df['DidFinish'] = cleaned_df['FinishPosition'].notna()
        cleaned_df['FinishedInPoints'] = (cleaned_df['FinishPosition'] <= 10) & cleaned_df['DidFinish']
        cleaned_df['FinishedOnPodium'] = (cleaned_df['FinishPosition'] <= 3) & cleaned_df['DidFinish']
        cleaned_df['Won'] = (cleaned_df['FinishPosition'] == 1)
        
        # 4. Handle lap time data
        time_columns = ['AvgLapTime', 'BestLapTime', 'AvgSector1', 'AvgSector2', 'AvgSector3']
        for col in time_columns:
            if col in cleaned_df.columns:
                # Remove extreme outliers (likely data errors)
                cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
                q99 = cleaned_df[col].quantile(0.99)
                q1 = cleaned_df[col].quantile(0.01)
                cleaned_df[col] = cleaned_df[col].clip(lower=q1, upper=q99)
        
        # 5. Clean consistency metric
        if 'Consistency' in cleaned_df.columns:
            cleaned_df['Consistency'] = pd.to_numeric(cleaned_df['Consistency'], errors='coerce')
            # Cap extreme consistency values
            cleaned_df['Consistency'] = cleaned_df['Consistency'].clip(upper=10.0)
        
        # 6. Standardize team and driver names
        cleaned_df['DriverName'] = cleaned_df['FullName'].str.strip()
        cleaned_df['TeamName'] = cleaned_df['TeamName'].str.strip()
        
        # Handle team name changes/variations
        team_mapping = {
            'AlphaTauri': 'AlphaTauri',
            'Red Bull Racing Honda RBPT': 'Red Bull Racing',
            'Red Bull Racing RBPT': 'Red Bull Racing',
            'Mercedes-AMG Petronas F1 Team': 'Mercedes',
            'Scuderia Ferrari': 'Ferrari',
            'McLaren F1 Team': 'McLaren',
            'Aston Martin Aramco Cognizant F1 Team': 'Aston Martin',
            'Alpine F1 Team': 'Alpine',
            'Williams Racing': 'Williams',
            'Alfa Romeo F1 Team Orlen': 'Alfa Romeo',
            'Haas F1 Team': 'Haas'
        }
        
        cleaned_df['TeamName'] = cleaned_df['TeamName'].map(team_mapping).fillna(cleaned_df['TeamName'])
        
        # 7. Create race metadata
        cleaned_df['RaceId'] = cleaned_df['Year'].astype(str) + '_' + cleaned_df['RaceName'].str.replace(' ', '_')
        
        print(f"✅ Data cleaning complete")
        print(f"📊 Rows: {original_rows} → {len(cleaned_df)} ({len(cleaned_df)/original_rows:.1%} retained)")
        
        return cleaned_df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create engineered features for machine learning"""
        print("⚙️ Engineering features...")
        
        featured_df = df.copy()
        
        # 1. Driver historical performance features
        featured_df = self._add_driver_features(featured_df)
        
        # 2. Team historical performance features  
        featured_df = self._add_team_features(featured_df)
        
        # 3. Circuit-specific features
        featured_df = self._add_circuit_features(featured_df)
        
        # 4. Recent form features
        featured_df = self._add_form_features(featured_df)
        
        # 5. Qualifying vs race performance
        featured_df = self._add_qualifying_features(featured_df)
        
        print("✅ Feature engineering complete")
        print(f"📊 Features created: {len([c for c in featured_df.columns if c not in df.columns])}")
        
        return featured_df
    
    def _add_driver_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add driver historical performance features"""
        
        # Calculate driver performance metrics up to each race
        driver_stats = []
        
        for race_id in df['RaceId'].unique():
            race_date = df[df['RaceId'] == race_id]['Date'].iloc[0]
            
            # Get historical data before this race
            historical = df[df['Date'] < race_date].copy()
            
            if historical.empty:
                continue
            
            # Calculate metrics for each driver
            driver_metrics = {}
            for driver in df['DriverName'].unique():
                driver_hist = historical[historical['DriverName'] == driver]
                
                if len(driver_hist) == 0:
                    driver_metrics[driver] = {
                        'DriverRaces': 0,
                        'DriverWins': 0,
                        'DriverPodiums': 0,
                        'DriverPoints': 0,
                        'DriverAvgPos': 15.0,
                        'DriverWinRate': 0.0,
                        'DriverPodiumRate': 0.0,
                        'DriverPointsRate': 0.0
                    }
                else:
                    races = len(driver_hist)
                    wins = driver_hist['Won'].sum()
                    podiums = driver_hist['FinishedOnPodium'].sum()
                    points_finishes = driver_hist['FinishedInPoints'].sum()
                    avg_pos = driver_hist['FinishPosition'].mean()
                    
                    driver_metrics[driver] = {
                        'DriverRaces': races,
                        'DriverWins': wins,
                        'DriverPodiums': podiums,
                        'DriverPoints': points_finishes,
                        'DriverAvgPos': avg_pos if not np.isnan(avg_pos) else 15.0,
                        'DriverWinRate': wins / races if races > 0 else 0.0,
                        'DriverPodiumRate': podiums / races if races > 0 else 0.0,
                        'DriverPointsRate': points_finishes / races if races > 0 else 0.0
                    }
            
            # Add to race data
            race_data = df[df['RaceId'] == race_id].copy()
            for idx, row in race_data.iterrows():
                driver = row['DriverName']
                if driver in driver_metrics:
                    for metric, value in driver_metrics[driver].items():
                        df.at[idx, metric] = value
        
        return df
    
    def _add_team_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add team historical performance features"""
        
        # Similar to driver features but for teams
        for race_id in df['RaceId'].unique():
            race_date = df[df['RaceId'] == race_id]['Date'].iloc[0]
            historical = df[df['Date'] < race_date].copy()
            
            if historical.empty:
                continue
            
            team_metrics = {}
            for team in df['TeamName'].unique():
                team_hist = historical[historical['TeamName'] == team]
                
                if len(team_hist) == 0:
                    team_metrics[team] = {
                        'TeamRaces': 0,
                        'TeamWins': 0,
                        'TeamPodiums': 0,
                        'TeamAvgPos': 15.0,
                        'TeamWinRate': 0.0,
                        'TeamPodiumRate': 0.0
                    }
                else:
                    races = len(team_hist)
                    wins = team_hist['Won'].sum()
                    podiums = team_hist['FinishedOnPodium'].sum()
                    avg_pos = team_hist['FinishPosition'].mean()
                    
                    team_metrics[team] = {
                        'TeamRaces': races,
                        'TeamWins': wins,
                        'TeamPodiums': podiums,
                        'TeamAvgPos': avg_pos if not np.isnan(avg_pos) else 15.0,
                        'TeamWinRate': wins / races if races > 0 else 0.0,
                        'TeamPodiumRate': podiums / races if races > 0 else 0.0
                    }
            
            # Add to race data
            race_data = df[df['RaceId'] == race_id].copy()
            for idx, row in race_data.iterrows():
                team = row['TeamName']
                if team in team_metrics:
                    for metric, value in team_metrics[team].items():
                        df.at[idx, metric] = value
        
        return df
    
    def _add_circuit_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add circuit-specific features"""
        
        # Circuit characteristics (could be enhanced with external data)
        circuit_features = {
            'Bahrain': {'Length': 5.412, 'Turns': 15, 'Type': 'Desert', 'Altitude': 7},
            'Saudi Arabia': {'Length': 6.174, 'Turns': 27, 'Type': 'Street', 'Altitude': 15},
            'Australia': {'Length': 5.278, 'Turns': 14, 'Type': 'Street', 'Altitude': 2},
            'Japan': {'Length': 5.807, 'Turns': 18, 'Type': 'Permanent', 'Altitude': 45},
            'China': {'Length': 5.451, 'Turns': 16, 'Type': 'Permanent', 'Altitude': 5},
            'Miami': {'Length': 5.41, 'Turns': 19, 'Type': 'Street', 'Altitude': 3},
            'Imola': {'Length': 4.909, 'Turns': 19, 'Type': 'Permanent', 'Altitude': 37},
            'Monaco': {'Length': 3.337, 'Turns': 19, 'Type': 'Street', 'Altitude': 7},
            'Canada': {'Length': 4.361, 'Turns': 14, 'Type': 'Permanent', 'Altitude': 20},
            'Spain': {'Length': 4.675, 'Turns': 16, 'Type': 'Permanent', 'Altitude': 109},
            'Austria': {'Length': 4.318, 'Turns': 10, 'Type': 'Permanent', 'Altitude': 678},
            'Britain': {'Length': 5.891, 'Turns': 18, 'Type': 'Permanent', 'Altitude': 153},
            'Hungary': {'Length': 4.381, 'Turns': 14, 'Type': 'Permanent', 'Altitude': 264},
            'Belgium': {'Length': 7.004, 'Turns': 19, 'Type': 'Permanent', 'Altitude': 401},
            'Netherlands': {'Length': 4.259, 'Turns': 14, 'Type': 'Permanent', 'Altitude': 6},
            'Italy': {'Length': 5.793, 'Turns': 11, 'Type': 'Permanent', 'Altitude': 162},
            'Singapore': {'Length': 5.063, 'Turns': 23, 'Type': 'Street', 'Altitude': 18},
            'Azerbaijan': {'Length': 6.003, 'Turns': 20, 'Type': 'Street', 'Altitude': -1},
            'United States': {'Length': 5.513, 'Turns': 20, 'Type': 'Permanent', 'Altitude': 161},
            'Mexico': {'Length': 4.304, 'Turns': 17, 'Type': 'Permanent', 'Altitude': 2285},
            'Brazil': {'Length': 4.309, 'Turns': 15, 'Type': 'Permanent', 'Altitude': 760},
            'Las Vegas': {'Length': 6.201, 'Turns': 17, 'Type': 'Street', 'Altitude': 610},
            'Qatar': {'Length': 5.419, 'Turns': 16, 'Type': 'Permanent', 'Altitude': 0},
            'Abu Dhabi': {'Length': 5.281, 'Turns': 16, 'Type': 'Permanent', 'Altitude': 3}
        }
        
        # Add circuit features
        for idx, row in df.iterrows():
            country = row.get('Country', '')
            if country in circuit_features:
                features = circuit_features[country]
                df.at[idx, 'CircuitLength'] = features['Length']
                df.at[idx, 'CircuitTurns'] = features['Turns']
                df.at[idx, 'CircuitAltitude'] = features['Altitude']
                df.at[idx, 'CircuitType'] = features['Type']
        
        return df
    
    def _add_form_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add recent form features (last N races)"""
        
        N_RACES = 5  # Look at last 5 races for form
        
        for race_id in df['RaceId'].unique():
            race_date = df[df['RaceId'] == race_id]['Date'].iloc[0]
            
            # Get recent races before this one
            recent_races = df[(df['Date'] < race_date)].copy()
            recent_races = recent_races.sort_values('Date', ascending=False)
            
            # Calculate form for each driver
            race_data = df[df['RaceId'] == race_id].copy()
            for idx, row in race_data.iterrows():
                driver = row['DriverName']
                driver_recent = recent_races[recent_races['DriverName'] == driver].head(N_RACES)
                
                if len(driver_recent) > 0:
                    # Recent form metrics
                    recent_avg_pos = driver_recent['FinishPosition'].mean()
                    recent_wins = driver_recent['Won'].sum()
                    recent_podiums = driver_recent['FinishedOnPodium'].sum()
                    recent_points = driver_recent['FinishedInPoints'].sum()
                    
                    df.at[idx, 'RecentAvgPos'] = recent_avg_pos if not np.isnan(recent_avg_pos) else 15.0
                    df.at[idx, 'RecentWins'] = recent_wins
                    df.at[idx, 'RecentPodiums'] = recent_podiums
                    df.at[idx, 'RecentPoints'] = recent_points
                    df.at[idx, 'RecentForm'] = len(driver_recent)
                else:
                    df.at[idx, 'RecentAvgPos'] = 15.0
                    df.at[idx, 'RecentWins'] = 0
                    df.at[idx, 'RecentPodiums'] = 0  
                    df.at[idx, 'RecentPoints'] = 0
                    df.at[idx, 'RecentForm'] = 0
        
        return df
    
    def _add_qualifying_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add qualifying position if available"""
        # This would require qualifying data - placeholder for now
        df['QualifyingPosition'] = np.random.randint(1, 21, size=len(df))  # Placeholder
        return df
    
    def prepare_for_ml(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Prepare data for machine learning"""
        print("🤖 Preparing data for machine learning...")
        
        ml_df = df.copy()
        
        # Select features for ML
        feature_columns = [
            # Driver features
            'DriverRaces', 'DriverWins', 'DriverPodiums', 'DriverAvgPos',
            'DriverWinRate', 'DriverPodiumRate', 'DriverPointsRate',
            
            # Team features  
            'TeamRaces', 'TeamWins', 'TeamPodiums', 'TeamAvgPos',
            'TeamWinRate', 'TeamPodiumRate',
            
            # Circuit features
            'CircuitLength', 'CircuitTurns', 'CircuitAltitude',
            
            # Form features
            'RecentAvgPos', 'RecentWins', 'RecentPodiums', 'RecentPoints', 'RecentForm',
            
            # Race features
            'QualifyingPosition'
        ]
        
        # Add lap time features if available
        lap_features = ['AvgLapTime', 'BestLapTime', 'Consistency']
        for feat in lap_features:
            if feat in ml_df.columns:
                feature_columns.append(feat)
        
        # Encode categorical variables
        if 'CircuitType' in ml_df.columns:
            le_circuit_type = LabelEncoder()
            ml_df['CircuitType_encoded'] = le_circuit_type.fit_transform(ml_df['CircuitType'].fillna('Unknown'))
            feature_columns.append('CircuitType_encoded')
            self.encoders['CircuitType'] = le_circuit_type
        
        # Handle missing values
        imputer = SimpleImputer(strategy='median')
        ml_df[feature_columns] = imputer.fit_transform(ml_df[feature_columns])
        
        # Scale features
        scaler = StandardScaler()
        ml_df[feature_columns] = scaler.fit_transform(ml_df[feature_columns])
        
        self.scalers['features'] = scaler
        self.feature_columns = feature_columns
        
        print(f"✅ ML preparation complete")
        print(f"📊 Features: {len(feature_columns)}")
        
        return ml_df, feature_columns
    
    def save_preprocessor(self, filename: str):
        """Save preprocessing components"""
        os.makedirs('models', exist_ok=True)
        
        preprocessor_data = {
            'scalers': self.scalers,
            'encoders': self.encoders,
            'feature_columns': self.feature_columns
        }
        
        import pickle
        with open(f'models/{filename}.pkl', 'wb') as f:
            pickle.dump(preprocessor_data, f)
        
        print(f"💾 Preprocessor saved to models/{filename}.pkl")
    
    def load_preprocessor(self, filename: str):
        """Load preprocessing components"""
        import pickle
        
        with open(f'models/{filename}.pkl', 'rb') as f:
            preprocessor_data = pickle.load(f)
        
        self.scalers = preprocessor_data['scalers']
        self.encoders = preprocessor_data['encoders']
        self.feature_columns = preprocessor_data['feature_columns']
        
        print(f"📂 Preprocessor loaded from models/{filename}.pkl")

def main():
    """Test the preprocessing pipeline"""
    from data_collection import F1DataCollector
    
    # Load or collect data
    collector = F1DataCollector()
    
    # Try to load existing data first
    data_files = [f for f in os.listdir('data') if f.startswith('f1_race_data') and f.endswith('.pkl')]
    
    if data_files:
        latest_file = sorted(data_files)[-1].replace('.pkl', '')
        df = collector.load_data(latest_file)
    else:
        print("No existing data found. Run data_collection.py first.")
        return
    
    if df.empty:
        print("❌ No data to process")
        return
    
    # Initialize preprocessor
    preprocessor = F1DataPreprocessor()
    
    # Clean data
    cleaned_df = preprocessor.clean_race_data(df)
    
    # Engineer features
    featured_df = preprocessor.engineer_features(cleaned_df)
    
    # Prepare for ML
    ml_df, features = preprocessor.prepare_for_ml(featured_df)
    
    # Save processed data
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    ml_df.to_pickle(f'data/processed_f1_data_{timestamp}.pkl')
    
    # Save preprocessor
    preprocessor.save_preprocessor(f'f1_preprocessor_{timestamp}')
    
    print(f"\n✅ Preprocessing complete!")
    print(f"📊 Final dataset shape: {ml_df.shape}")
    print(f"📈 Features: {len(features)}")
    
    # Show sample of Mexico data
    mexico_data = ml_df[ml_df['Country'] == 'Mexico']
    if not mexico_data.empty:
        print(f"\n🇲🇽 Mexico GP processed data: {len(mexico_data)} entries")
        print("Recent Mexico results:")
        for year in sorted(mexico_data['Year'].unique(), reverse=True):
            year_data = mexico_data[mexico_data['Year'] == year].sort_values('FinishPosition')
            if not year_data.empty:
                winner = year_data.iloc[0]
                print(f"  {year}: {winner['DriverName']} ({winner['TeamName']})")

if __name__ == "__main__":
    main()