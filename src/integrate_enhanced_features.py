#!/usr/bin/env python3
"""
Integration Script for Enhanced OpenF1 Features
Shows how to integrate advanced OpenF1 features into your existing prediction model
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import warnings
warnings.filterwarnings('ignore')

from enhanced_openf1_features import EnhancedOpenF1Extractor

class F1ModelEnhancer:
    
    def __init__(self):
        self.extractor = EnhancedOpenF1Extractor()
        self.encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = []
        
    def collect_historical_data_with_features(self, sessions_list: list) -> pd.DataFrame:
        """Collect historical data with enhanced features"""
        print("🔄 Collecting historical data with enhanced features...")
        
        all_data = []
        
        for session_info in sessions_list:
            session_key = session_info['session_key']
            meeting_key = session_info['meeting_key']
            circuit = session_info.get('circuit_short_name', 'Unknown')
            year = session_info.get('year', 2024)
            
            print(f"Processing: {circuit} {year} - Session {session_key}")
            
            # Extract enhanced features
            features = self.extractor.extract_comprehensive_features(session_key, meeting_key)
            
            # Get drivers info
            drivers = self.extractor.safe_api_call("drivers", {"session_key": session_key})
            
            if drivers and features:
                # Create feature DataFrame
                df_session = self.extractor.create_feature_dataframe(features, drivers)
                
                # Add session metadata
                df_session['session_key'] = session_key
                df_session['meeting_key'] = meeting_key
                df_session['circuit'] = circuit
                df_session['year'] = year
                df_session['session_type'] = session_info.get('session_type', 'Unknown')
                
                # Add target variable (you'll need to get actual results)
                df_session['target'] = self.get_target_variable(session_key, df_session)
                
                all_data.append(df_session)
                
                # Rate limiting
                import time
                time.sleep(0.5)
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            return combined_df
        else:
            return pd.DataFrame()
    
    def get_target_variable(self, session_key: str, df_session: pd.DataFrame) -> list:
        """Get target variable (top 5 finish) for each driver"""
        # Get race results
        results = self.extractor.safe_api_call("session_result", {"session_key": session_key})
        
        if not results:
            # Return default targets if no results available
            return [0] * len(df_session)
        
        # Create target mapping
        target_map = {}
        for result in results:
            driver_num = result['driver_number']
            position = result.get('position', 21)
            
            try:
                position = int(position)
                target_map[driver_num] = 1 if position <= 5 else 0
            except (ValueError, TypeError):
                target_map[driver_num] = 0
        
        # Map to DataFrame
        targets = []
        for _, row in df_session.iterrows():
            driver_num = row['driver_number']
            targets.append(target_map.get(driver_num, 0))
        
        return targets
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for model training"""
        print("🔧 Preparing features for training...")
        
        # Define feature categories
        numerical_features = [
            'practice_avg_lap', 'practice_best_lap', 'practice_consistency', 'practice_pace_rank',
            'quali_best_lap', 'quali_sector_1', 'quali_sector_2', 'quali_sector_3',
            'avg_speed', 'max_speed', 'throttle_efficiency', 'brake_efficiency', 'gear_changes',
            'race_avg_pace', 'race_consistency', 'pace_degradation', 'net_position_change',
            'pit_count', 'avg_pit_duration', 'weather_stability', 'rain_probability', 'temp_variance'
        ]
        
        categorical_features = [
            'team_name', 'country_code', 'circuit', 'pit_strategy_type'
        ]
        
        # Create feature matrix
        feature_df = df.copy()
        
        # Fill missing values
        for col in numerical_features:
            if col in feature_df.columns:
                feature_df[col] = feature_df[col].fillna(feature_df[col].median())
            else:
                feature_df[col] = 0
        
        for col in categorical_features:
            if col in feature_df.columns:
                feature_df[col] = feature_df[col].fillna('Unknown')
            else:
                feature_df[col] = 'Unknown'
        
        # Encode categorical features
        for col in categorical_features:
            if col not in self.encoders:
                self.encoders[col] = LabelEncoder()
                feature_df[col] = self.encoders[col].fit_transform(feature_df[col].astype(str))
            else:
                # Handle new categories
                feature_df[col] = feature_df[col].astype(str)
                known_categories = set(self.encoders[col].classes_)
                feature_df[col] = feature_df[col].apply(
                    lambda x: x if x in known_categories else self.encoders[col].classes_[0]
                )
                feature_df[col] = self.encoders[col].transform(feature_df[col])\
        
        # Create additional engineered features
        feature_df = self.create_engineered_features(feature_df)\n        
        # Store feature columns
        self.feature_columns = numerical_features + categorical_features + ['pace_advantage', 'consistency_score', 'strategy_score']
        
        return feature_df[self.feature_columns]
    
    def create_engineered_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create additional engineered features"""
        
        # Pace advantage (compared to session average)
        if 'practice_avg_lap' in df.columns:
            session_avg_pace = df.groupby('session_key')['practice_avg_lap'].transform('mean')
            df['pace_advantage'] = (session_avg_pace - df['practice_avg_lap']) / session_avg_pace
        else:
            df['pace_advantage'] = 0
        
        # Consistency score (lower is better)
        consistency_cols = ['practice_consistency', 'race_consistency', 'throttle_efficiency', 'brake_efficiency']
        consistency_values = []
        for col in consistency_cols:
            if col in df.columns:
                consistency_values.append(df[col].fillna(0))
        
        if consistency_values:
            df['consistency_score'] = np.mean(consistency_values, axis=0)
        else:
            df['consistency_score'] = 0
        
        # Strategy score (pit strategy effectiveness)
        df['strategy_score'] = 0
        if 'pit_count' in df.columns and 'avg_pit_duration' in df.columns:
            # Normalize pit count and duration
            df['strategy_score'] = (df['pit_count'] * 0.3) + (df['avg_pit_duration'] * 0.7)
        
        return df
    
    def train_enhanced_model(self, X: pd.DataFrame, y: pd.Series) -> RandomForestClassifier:
        """Train enhanced model with new features"""
        print("🤖 Training enhanced model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"✅ Model trained successfully!")
        print(f"Accuracy: {accuracy:.3f}")
        print(f"Features used: {len(self.feature_columns)}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        print(feature_importance.head(10))
        
        return model
    
    def save_enhanced_model(self, model, filepath: str):
        """Save the enhanced model and preprocessing components"""
        print(f"💾 Saving enhanced model to {filepath}")
        
        model_package = {
            'model': model,
            'scaler': self.scaler,
            'encoders': self.encoders,
            'feature_columns': self.feature_columns
        }
        
        joblib.dump(model_package, filepath)
        print("✅ Model saved successfully!")
    
    def load_enhanced_model(self, filepath: str):
        """Load the enhanced model and preprocessing components"""
        print(f"📂 Loading enhanced model from {filepath}")
        
        model_package = joblib.load(filepath)
        
        model = model_package['model']
        self.scaler = model_package['scaler']
        self.encoders = model_package['encoders']
        self.feature_columns = model_package['feature_columns']
        
        print("✅ Model loaded successfully!")
        return model
    
    def predict_with_enhanced_features(self, session_key: str, meeting_key: str, model) -> pd.DataFrame:
        """Make predictions using enhanced features"""
        print(f"🔮 Making predictions with enhanced features for session {session_key}")
        
        # Extract features for the session
        features = self.extractor.extract_comprehensive_features(session_key, meeting_key)
        
        # Get drivers info
        drivers = self.extractor.safe_api_call("drivers", {"session_key": session_key})
        
        if not drivers or not features:
            print("❌ Could not extract features for prediction")
            return pd.DataFrame()
        
        # Create feature DataFrame
        df_session = self.extractor.create_feature_dataframe(features, drivers)
        
        # Prepare features
        X = self.prepare_features(df_session)
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        predictions = model.predict_proba(X_scaled)[:, 1]  # Probability of top 5
        
        # Create results DataFrame
        results = pd.DataFrame({
            'driver_name': df_session['driver_name'],
            'team_name': df_session['team_name'],
            'driver_number': df_session['driver_number'],
            'top5_probability': predictions
        })
        
        # Sort by probability
        results = results.sort_values('top5_probability', ascending=False)
        
        return results

def main():
    """Example usage of enhanced F1 model"""
    print("🚀 Starting Enhanced F1 Model Integration...")
    
    enhancer = F1ModelEnhancer()
    
    # Example: You would get this from your existing session data
    sample_sessions = [
        {
            'session_key': '9158',
            'meeting_key': '1217',
            'circuit_short_name': 'Silverstone',
            'year': 2024,
            'session_type': 'Race'
        }
        # Add more sessions for training
    ]
    
    print("📊 This is a demo of the enhanced feature extraction system.")
    print("To use this with your model:")
    print("1. Collect historical session data")
    print("2. Extract enhanced features using the EnhancedOpenF1Extractor")
    print("3. Train your model with the additional features")
    print("4. Use the enhanced model for predictions")
    
    # Demo feature extraction
    if sample_sessions:
        session = sample_sessions[0]
        features = enhancer.extractor.extract_comprehensive_features(
            session['session_key'], 
            session['meeting_key']
        )
        
        if features:
            print(f"\n✅ Successfully extracted features:")
            for feature_type, data in features.items():
                print(f"  - {feature_type}: {len(data) if isinstance(data, dict) else 'N/A'} drivers")
        else:
            print("❌ Could not extract features (possibly due to API limits)")

if __name__ == "__main__":
    main()
