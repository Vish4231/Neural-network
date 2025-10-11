#!/usr/bin/env python3
"""
Pattern Recognition Engine for F1 Predictions
Phase 2: Identify probabilities and trends from historical data
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class F1PatternRecognitionEngine:
    def __init__(self, data_path: str = 'data/integrated_f1_data.csv'):
        self.data_path = data_path
        self.data = None
        self.patterns = {}
        self.probabilities = {}
        self.trends = {}
        
        print("🧠 F1 Pattern Recognition Engine Initialized")
    
    def load_data(self) -> pd.DataFrame:
        """Load and prepare data for pattern recognition"""
        try:
            self.data = pd.read_csv(self.data_path)
            print(f"📊 Loaded {len(self.data)} records for pattern analysis")
            return self.data
        except FileNotFoundError:
            print(f"❌ Data file not found: {self.data_path}")
            print("Please run the data pipeline first")
            return pd.DataFrame()
    
    def analyze_driver_patterns(self) -> Dict:
        """
        Analyze driver-specific patterns and probabilities
        """
        print("🔍 Analyzing driver patterns...")
        
        if self.data is None:
            self.load_data()
        
        driver_patterns = {}
        
        for driver in self.data['driver_name'].unique():
            if pd.isna(driver) or driver == 'Unknown':
                continue
                
            driver_data = self.data[self.data['driver_name'] == driver].copy()
            
            if len(driver_data) < 5:  # Need minimum races for pattern analysis
                continue
            
            # Calculate driver statistics
            patterns = {
                'total_races': len(driver_data),
                'avg_finish': driver_data['finishing_position'].mean(),
                'median_finish': driver_data['finishing_position'].median(),
                'std_finish': driver_data['finishing_position'].std(),
                'podium_rate': (driver_data['finishing_position'] <= 3).mean(),
                'top5_rate': (driver_data['finishing_position'] <= 5).mean(),
                'points_rate': (driver_data['finishing_position'] <= 10).mean(),
                'dnf_rate': (driver_data['finishing_position'] > 20).mean(),
                'qualifying_avg': driver_data['qualifying_position'].mean() if 'qualifying_position' in driver_data.columns else np.nan,
                'grid_to_finish_avg': (driver_data['grid_position'] - driver_data['finishing_position']).mean() if 'grid_position' in driver_data.columns else np.nan,
                'consistency_score': 1 / (1 + driver_data['finishing_position'].std()),  # Higher = more consistent
                'improvement_trend': self._calculate_improvement_trend(driver_data),
                'circuit_specialist': self._identify_circuit_specialist(driver_data),
                'weather_performance': self._analyze_weather_performance(driver_data),
                'recent_form': self._calculate_recent_form(driver_data)
            }
            
            driver_patterns[driver] = patterns
        
        self.patterns['drivers'] = driver_patterns
        print(f"✅ Analyzed patterns for {len(driver_patterns)} drivers")
        return driver_patterns
    
    def analyze_team_patterns(self) -> Dict:
        """
        Analyze team-specific patterns and probabilities
        """
        print("🔍 Analyzing team patterns...")
        
        if self.data is None:
            self.load_data()
        
        team_patterns = {}
        
        for team in self.data['team_name'].unique():
            if pd.isna(team) or team == 'Unknown':
                continue
                
            team_data = self.data[self.data['team_name'] == team].copy()
            
            if len(team_data) < 10:  # Need minimum races for team analysis
                continue
            
            # Calculate team statistics
            patterns = {
                'total_races': len(team_data),
                'avg_finish': team_data['finishing_position'].mean(),
                'median_finish': team_data['finishing_position'].median(),
                'std_finish': team_data['finishing_position'].std(),
                'podium_rate': (team_data['finishing_position'] <= 3).mean(),
                'top5_rate': (team_data['finishing_position'] <= 5).mean(),
                'points_rate': (team_data['finishing_position'] <= 10).mean(),
                'dnf_rate': (team_data['finishing_position'] > 20).mean(),
                'qualifying_avg': team_data['qualifying_position'].mean() if 'qualifying_position' in team_data.columns else np.nan,
                'team_consistency': 1 / (1 + team_data.groupby('year')['finishing_position'].mean().std()),
                'development_trend': self._calculate_team_development_trend(team_data),
                'circuit_performance': self._analyze_team_circuit_performance(team_data),
                'reliability_score': 1 - team_data['dnf_rate'],
                'championship_pressure': self._calculate_championship_pressure(team_data)
            }
            
            team_patterns[team] = patterns
        
        self.patterns['teams'] = team_patterns
        print(f"✅ Analyzed patterns for {len(team_patterns)} teams")
        return team_patterns
    
    def analyze_circuit_patterns(self) -> Dict:
        """
        Analyze circuit-specific patterns and probabilities
        """
        print("🔍 Analyzing circuit patterns...")
        
        if self.data is None:
            self.load_data()
        
        circuit_patterns = {}
        
        for circuit in self.data['circuit'].unique():
            if pd.isna(circuit) or circuit == 'Unknown':
                continue
                
            circuit_data = self.data[self.data['circuit'] == circuit].copy()
            
            if len(circuit_data) < 5:  # Need minimum races for circuit analysis
                continue
            
            # Calculate circuit statistics
            patterns = {
                'total_races': len(circuit_data),
                'avg_finish_variance': circuit_data['finishing_position'].std(),
                'overtaking_opportunities': self._calculate_overtaking_opportunities(circuit_data),
                'safety_car_probability': self._calculate_safety_car_probability(circuit_data),
                'weather_impact': self._analyze_circuit_weather_impact(circuit_data),
                'grid_position_importance': self._calculate_grid_importance(circuit_data),
                'tire_degradation_impact': self._analyze_tire_degradation(circuit_data),
                'qualifying_correlation': self._calculate_qualifying_correlation(circuit_data),
                'driver_specialists': self._identify_circuit_specialists(circuit_data),
                'team_advantages': self._identify_team_advantages(circuit_data)
            }
            
            circuit_patterns[circuit] = patterns
        
        self.patterns['circuits'] = circuit_patterns
        print(f"✅ Analyzed patterns for {len(circuit_patterns)} circuits")
        return circuit_patterns
    
    def calculate_position_probabilities(self) -> Dict:
        """
        Calculate probability distributions for different finishing positions
        """
        print("🎯 Calculating position probabilities...")
        
        if self.data is None:
            self.load_data()
        
        probabilities = {}
        
        # Overall position probabilities
        position_counts = self.data['finishing_position'].value_counts().sort_index()
        total_races = len(self.data)
        probabilities['overall'] = (position_counts / total_races).to_dict()
        
        # Driver-specific position probabilities
        driver_probs = {}
        for driver in self.data['driver_name'].unique():
            if pd.isna(driver) or driver == 'Unknown':
                continue
                
            driver_data = self.data[self.data['driver_name'] == driver]
            if len(driver_data) >= 5:
                driver_positions = driver_data['finishing_position'].value_counts().sort_index()
                driver_probs[driver] = (driver_positions / len(driver_data)).to_dict()
        
        probabilities['drivers'] = driver_probs
        
        # Team-specific position probabilities
        team_probs = {}
        for team in self.data['team_name'].unique():
            if pd.isna(team) or team == 'Unknown':
                continue
                
            team_data = self.data[self.data['team_name'] == team]
            if len(team_data) >= 10:
                team_positions = team_data['finishing_position'].value_counts().sort_index()
                team_probs[team] = (team_positions / len(team_data)).to_dict()
        
        probabilities['teams'] = team_probs
        
        # Circuit-specific position probabilities
        circuit_probs = {}
        for circuit in self.data['circuit'].unique():
            if pd.isna(circuit) or circuit == 'Unknown':
                continue
                
            circuit_data = self.data[self.data['circuit'] == circuit]
            if len(circuit_data) >= 5:
                circuit_positions = circuit_data['finishing_position'].value_counts().sort_index()
                circuit_probs[circuit] = (circuit_positions / len(circuit_data)).to_dict()
        
        probabilities['circuits'] = circuit_probs
        
        self.probabilities = probabilities
        print("✅ Position probabilities calculated")
        return probabilities
    
    def identify_trends(self) -> Dict:
        """
        Identify temporal trends in performance
        """
        print("📈 Identifying performance trends...")
        
        if self.data is None:
            self.load_data()
        
        trends = {}
        
        # Overall performance trends
        yearly_avg = self.data.groupby('year')['finishing_position'].mean()
        trends['overall'] = {
            'yearly_avg_finish': yearly_avg.to_dict(),
            'performance_trend': self._calculate_trend_slope(yearly_avg.values),
            'volatility_trend': self._calculate_volatility_trend(yearly_avg.values)
        }
        
        # Driver performance trends
        driver_trends = {}
        for driver in self.data['driver_name'].unique():
            if pd.isna(driver) or driver == 'Unknown':
                continue
                
            driver_data = self.data[self.data['driver_name'] == driver]
            if len(driver_data) >= 10:
                yearly_performance = driver_data.groupby('year')['finishing_position'].mean()
                if len(yearly_performance) >= 3:
                    driver_trends[driver] = {
                        'yearly_performance': yearly_performance.to_dict(),
                        'trend_slope': self._calculate_trend_slope(yearly_performance.values),
                        'consistency_trend': self._calculate_consistency_trend(driver_data),
                        'peak_performance_year': yearly_performance.idxmin(),
                        'decline_indicator': self._detect_performance_decline(yearly_performance)
                    }
        
        trends['drivers'] = driver_trends
        
        # Team performance trends
        team_trends = {}
        for team in self.data['team_name'].unique():
            if pd.isna(team) or team == 'Unknown':
                continue
                
            team_data = self.data[self.data['team_name'] == team]
            if len(team_data) >= 20:
                yearly_performance = team_data.groupby('year')['finishing_position'].mean()
                if len(yearly_performance) >= 3:
                    team_trends[team] = {
                        'yearly_performance': yearly_performance.to_dict(),
                        'trend_slope': self._calculate_trend_slope(yearly_performance.values),
                        'development_phase': self._identify_development_phase(yearly_performance),
                        'championship_contention': self._assess_championship_contention(team_data)
                    }
        
        trends['teams'] = team_trends
        
        self.trends = trends
        print("✅ Performance trends identified")
        return trends
    
    def cluster_analysis(self) -> Dict:
        """
        Perform clustering analysis to identify driver/team groups
        """
        print("🔬 Performing clustering analysis...")
        
        if self.data is None:
            self.load_data()
        
        # Prepare features for clustering
        features = self._prepare_clustering_features()
        
        if features.empty:
            print("⚠️ No features available for clustering")
            return {}
        
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features.select_dtypes(include=[np.number]))
        
        # Perform K-means clustering
        n_clusters = min(5, len(features))  # Max 5 clusters
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(features_scaled)
        
        # Add cluster labels to features
        features['cluster'] = clusters
        
        # Analyze clusters
        cluster_analysis = {}
        for cluster_id in range(n_clusters):
            cluster_data = features[features['cluster'] == cluster_id]
            cluster_analysis[f'cluster_{cluster_id}'] = {
                'size': len(cluster_data),
                'avg_performance': cluster_data['avg_finish'].mean() if 'avg_finish' in cluster_data.columns else np.nan,
                'consistency': cluster_data['consistency_score'].mean() if 'consistency_score' in cluster_data.columns else np.nan,
                'members': cluster_data.index.tolist()
            }
        
        print(f"✅ Clustering analysis complete: {n_clusters} clusters identified")
        return cluster_analysis
    
    def generate_prediction_features(self) -> pd.DataFrame:
        """
        Generate features for prediction based on identified patterns
        """
        print("🎯 Generating prediction features...")
        
        if not self.patterns or not self.probabilities or not self.trends:
            print("⚠️ Patterns, probabilities, and trends must be calculated first")
            return pd.DataFrame()
        
        # Create feature matrix
        features = []
        
        for _, row in self.data.iterrows():
            driver = row['driver_name']
            team = row['team_name']
            circuit = row['circuit']
            year = row['year']
            
            # Driver features
            driver_features = self.patterns['drivers'].get(driver, {})
            driver_probs = self.probabilities['drivers'].get(driver, {})
            driver_trends = self.trends['drivers'].get(driver, {})
            
            # Team features
            team_features = self.patterns['teams'].get(team, {})
            team_probs = self.probabilities['teams'].get(team, {})
            team_trends = self.trends['teams'].get(team, {})
            
            # Circuit features
            circuit_features = self.patterns['circuits'].get(circuit, {})
            circuit_probs = self.probabilities['circuits'].get(circuit, {})
            
            # Combine all features
            feature_row = {
                'driver_name': driver,
                'team_name': team,
                'circuit': circuit,
                'year': year,
                'finishing_position': row['finishing_position'],
                # Driver features
                'driver_avg_finish': driver_features.get('avg_finish', 10),
                'driver_consistency': driver_features.get('consistency_score', 0.5),
                'driver_podium_rate': driver_features.get('podium_rate', 0.1),
                'driver_top5_rate': driver_features.get('top5_rate', 0.2),
                'driver_recent_form': driver_features.get('recent_form', 10),
                'driver_trend_slope': driver_trends.get('trend_slope', 0),
                # Team features
                'team_avg_finish': team_features.get('avg_finish', 10),
                'team_consistency': team_features.get('team_consistency', 0.5),
                'team_podium_rate': team_features.get('podium_rate', 0.1),
                'team_top5_rate': team_features.get('top5_rate', 0.2),
                'team_development_trend': team_features.get('development_trend', 0),
                'team_reliability': team_features.get('reliability_score', 0.8),
                # Circuit features
                'circuit_overtaking': circuit_features.get('overtaking_opportunities', 0.5),
                'circuit_safety_car_prob': circuit_features.get('safety_car_probability', 0.3),
                'circuit_grid_importance': circuit_features.get('grid_position_importance', 0.5),
                'circuit_qualifying_correlation': circuit_features.get('qualifying_correlation', 0.7)
            }
            
            features.append(feature_row)
        
        features_df = pd.DataFrame(features)
        print(f"✅ Generated {len(features_df)} prediction features")
        return features_df
    
    # Helper methods
    def _calculate_improvement_trend(self, driver_data: pd.DataFrame) -> float:
        """Calculate if driver is improving over time"""
        if len(driver_data) < 5:
            return 0.0
        
        yearly_avg = driver_data.groupby('year')['finishing_position'].mean()
        if len(yearly_avg) < 3:
            return 0.0
        
        # Negative slope means improving (lower position numbers)
        slope, _ = stats.linregress(range(len(yearly_avg)), yearly_avg.values)
        return -slope  # Negative because lower positions are better
    
    def _identify_circuit_specialist(self, driver_data: pd.DataFrame) -> str:
        """Identify driver's best circuit"""
        if len(driver_data) < 5:
            return 'Unknown'
        
        circuit_avg = driver_data.groupby('circuit')['finishing_position'].mean()
        return circuit_avg.idxmin() if not circuit_avg.empty else 'Unknown'
    
    def _analyze_weather_performance(self, driver_data: pd.DataFrame) -> Dict:
        """Analyze driver performance in different weather conditions"""
        # Simplified weather analysis
        return {
            'dry_performance': driver_data['finishing_position'].mean(),
            'wet_performance': driver_data['finishing_position'].mean()  # Placeholder
        }
    
    def _calculate_recent_form(self, driver_data: pd.DataFrame, races: int = 5) -> float:
        """Calculate recent form (last N races)"""
        if len(driver_data) < races:
            return driver_data['finishing_position'].mean()
        
        recent_data = driver_data.tail(races)
        return recent_data['finishing_position'].mean()
    
    def _calculate_team_development_trend(self, team_data: pd.DataFrame) -> float:
        """Calculate team development trend"""
        if len(team_data) < 10:
            return 0.0
        
        yearly_avg = team_data.groupby('year')['finishing_position'].mean()
        if len(yearly_avg) < 3:
            return 0.0
        
        slope, _ = stats.linregress(range(len(yearly_avg)), yearly_avg.values)
        return -slope  # Negative because lower positions are better
    
    def _analyze_team_circuit_performance(self, team_data: pd.DataFrame) -> Dict:
        """Analyze team performance at different circuits"""
        circuit_avg = team_data.groupby('circuit')['finishing_position'].mean()
        return {
            'best_circuit': circuit_avg.idxmin() if not circuit_avg.empty else 'Unknown',
            'worst_circuit': circuit_avg.idxmax() if not circuit_avg.empty else 'Unknown',
            'circuit_consistency': 1 / (1 + circuit_avg.std())
        }
    
    def _calculate_championship_pressure(self, team_data: pd.DataFrame) -> float:
        """Calculate championship pressure based on recent performance"""
        if len(team_data) < 5:
            return 0.5
        
        recent_performance = team_data.tail(10)['finishing_position'].mean()
        # Higher pressure for teams performing well
        return max(0, min(1, (10 - recent_performance) / 10))
    
    def _calculate_overtaking_opportunities(self, circuit_data: pd.DataFrame) -> float:
        """Calculate overtaking opportunities at circuit"""
        if 'grid_position' in circuit_data.columns and 'finishing_position' in circuit_data.columns:
            position_changes = (circuit_data['grid_position'] - circuit_data['finishing_position']).abs().mean()
            return min(1.0, position_changes / 10)  # Normalize to 0-1
        return 0.5
    
    def _calculate_safety_car_probability(self, circuit_data: pd.DataFrame) -> float:
        """Calculate safety car probability at circuit"""
        # Simplified calculation based on DNF rate
        dnf_rate = (circuit_data['finishing_position'] > 20).mean()
        return min(1.0, dnf_rate * 2)  # Higher DNF rate = higher SC probability
    
    def _analyze_circuit_weather_impact(self, circuit_data: pd.DataFrame) -> float:
        """Analyze weather impact on circuit"""
        # Placeholder - would need weather data
        return 0.5
    
    def _calculate_grid_importance(self, circuit_data: pd.DataFrame) -> float:
        """Calculate importance of grid position at circuit"""
        if 'grid_position' in circuit_data.columns and 'finishing_position' in circuit_data.columns:
            correlation = circuit_data['grid_position'].corr(circuit_data['finishing_position'])
            return abs(correlation) if not pd.isna(correlation) else 0.5
        return 0.5
    
    def _analyze_tire_degradation(self, circuit_data: pd.DataFrame) -> float:
        """Analyze tire degradation impact at circuit"""
        # Placeholder - would need tire data
        return 0.5
    
    def _calculate_qualifying_correlation(self, circuit_data: pd.DataFrame) -> float:
        """Calculate correlation between qualifying and race results"""
        if 'qualifying_position' in circuit_data.columns and 'finishing_position' in circuit_data.columns:
            correlation = circuit_data['qualifying_position'].corr(circuit_data['finishing_position'])
            return abs(correlation) if not pd.isna(correlation) else 0.7
        return 0.7
    
    def _identify_circuit_specialists(self, circuit_data: pd.DataFrame) -> List[str]:
        """Identify drivers who perform well at this circuit"""
        driver_avg = circuit_data.groupby('driver_name')['finishing_position'].mean()
        return driver_avg.nsmallest(3).index.tolist()
    
    def _identify_team_advantages(self, circuit_data: pd.DataFrame) -> List[str]:
        """Identify teams with advantages at this circuit"""
        team_avg = circuit_data.groupby('team_name')['finishing_position'].mean()
        return team_avg.nsmallest(3).index.tolist()
    
    def _calculate_trend_slope(self, values: np.ndarray) -> float:
        """Calculate slope of trend line"""
        if len(values) < 2:
            return 0.0
        
        x = np.arange(len(values))
        slope, _ = stats.linregress(x, values)
        return slope
    
    def _calculate_volatility_trend(self, values: np.ndarray) -> float:
        """Calculate trend in volatility"""
        if len(values) < 3:
            return 0.0
        
        # Calculate rolling standard deviation
        rolling_std = pd.Series(values).rolling(window=3, min_periods=2).std()
        if len(rolling_std.dropna()) < 2:
            return 0.0
        
        slope, _ = stats.linregress(range(len(rolling_std.dropna())), rolling_std.dropna().values)
        return slope
    
    def _calculate_consistency_trend(self, driver_data: pd.DataFrame) -> float:
        """Calculate trend in driver consistency"""
        if len(driver_data) < 10:
            return 0.0
        
        yearly_std = driver_data.groupby('year')['finishing_position'].std()
        if len(yearly_std) < 3:
            return 0.0
        
        slope, _ = stats.linregress(range(len(yearly_std)), yearly_std.values)
        return -slope  # Negative because lower std is better consistency
    
    def _detect_performance_decline(self, yearly_performance: pd.Series) -> bool:
        """Detect if driver is in performance decline"""
        if len(yearly_performance) < 3:
            return False
        
        # Check if recent performance is significantly worse than peak
        peak_performance = yearly_performance.min()
        recent_performance = yearly_performance.tail(2).mean()
        
        return recent_performance > peak_performance + 2  # Significant decline
    
    def _identify_development_phase(self, yearly_performance: pd.Series) -> str:
        """Identify team development phase"""
        if len(yearly_performance) < 3:
            return 'Unknown'
        
        slope, _ = stats.linregress(range(len(yearly_performance)), yearly_performance.values)
        
        if slope < -0.5:
            return 'Improving'
        elif slope > 0.5:
            return 'Declining'
        else:
            return 'Stable'
    
    def _assess_championship_contention(self, team_data: pd.DataFrame) -> str:
        """Assess team's championship contention level"""
        if len(team_data) < 10:
            return 'Unknown'
        
        recent_performance = team_data.tail(10)['finishing_position'].mean()
        
        if recent_performance <= 3:
            return 'Championship Contender'
        elif recent_performance <= 6:
            return 'Podium Contender'
        elif recent_performance <= 10:
            return 'Points Contender'
        else:
            return 'Backmarker'
    
    def _prepare_clustering_features(self) -> pd.DataFrame:
        """Prepare features for clustering analysis"""
        if not self.patterns:
            return pd.DataFrame()
        
        features = []
        
        # Driver features
        for driver, patterns in self.patterns['drivers'].items():
            features.append({
                'name': driver,
                'type': 'driver',
                'avg_finish': patterns.get('avg_finish', 10),
                'consistency_score': patterns.get('consistency_score', 0.5),
                'podium_rate': patterns.get('podium_rate', 0.1),
                'top5_rate': patterns.get('top5_rate', 0.2)
            })
        
        # Team features
        for team, patterns in self.patterns['teams'].items():
            features.append({
                'name': team,
                'type': 'team',
                'avg_finish': patterns.get('avg_finish', 10),
                'consistency_score': patterns.get('team_consistency', 0.5),
                'podium_rate': patterns.get('podium_rate', 0.1),
                'top5_rate': patterns.get('top5_rate', 0.2)
            })
        
        return pd.DataFrame(features)

def main():
    """Main function to run pattern recognition analysis"""
    print("🧠 Starting F1 Pattern Recognition Analysis")
    print("=" * 50)
    
    # Initialize pattern recognition engine
    engine = F1PatternRecognitionEngine()
    
    # Load data
    data = engine.load_data()
    if data.empty:
        print("❌ No data available for pattern recognition")
        return
    
    # Run all analyses
    print("\n🔍 Running comprehensive pattern analysis...")
    
    # Analyze patterns
    driver_patterns = engine.analyze_driver_patterns()
    team_patterns = engine.analyze_team_patterns()
    circuit_patterns = engine.analyze_circuit_patterns()
    
    # Calculate probabilities
    probabilities = engine.calculate_position_probabilities()
    
    # Identify trends
    trends = engine.identify_trends()
    
    # Perform clustering
    clusters = engine.cluster_analysis()
    
    # Generate prediction features
    prediction_features = engine.generate_prediction_features()
    
    # Save results
    output_dir = 'data/pattern_analysis'
    os.makedirs(output_dir, exist_ok=True)
    
    prediction_features.to_csv(f'{output_dir}/prediction_features.csv', index=False)
    
    print("\n✅ Pattern Recognition Analysis Complete!")
    print("=" * 50)
    print("Results saved to:")
    print(f"  - Prediction features: {output_dir}/prediction_features.csv")
    print("\nNext steps:")
    print("1. Use prediction features for model training")
    print("2. Implement prediction system")
    print("3. Validate prediction accuracy")

if __name__ == "__main__":
    main()
