#!/usr/bin/env python3
"""
Advanced F1 Prediction System
Phase 3: Accurate future race prediction using test data and actual data
"""

import pandas as pd
import numpy as np
import joblib
import os
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

class AdvancedF1PredictionSystem:
    def __init__(self, data_path: str = 'data/pattern_analysis/prediction_features.csv'):
        self.data_path = data_path
        self.data = None
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_importance = {}
        self.prediction_accuracy = {}
        
        print("🎯 Advanced F1 Prediction System Initialized")
    
    def load_data(self) -> pd.DataFrame:
        """Load prediction features data"""
        try:
            self.data = pd.read_csv(self.data_path)
            print(f"📊 Loaded {len(self.data)} records for prediction training")
            return self.data
        except FileNotFoundError:
            print(f"❌ Data file not found: {self.data_path}")
            print("Please run pattern recognition analysis first")
            return pd.DataFrame()
    
    def prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare data for model training
        """
        if self.data is None:
            self.load_data()
        
        if self.data.empty:
            print("❌ No data available for training")
            return np.array([]), np.array([]), []
        
        print("🔧 Preparing training data...")
        
        # Select features for training
        feature_columns = [
            'driver_avg_finish', 'driver_consistency', 'driver_podium_rate', 'driver_top5_rate',
            'driver_recent_form', 'driver_trend_slope', 'team_avg_finish', 'team_consistency',
            'team_podium_rate', 'team_top5_rate', 'team_development_trend', 'team_reliability',
            'circuit_overtaking', 'circuit_safety_car_prob', 'circuit_grid_importance',
            'circuit_qualifying_correlation'
        ]
        
        # Filter available features
        available_features = [col for col in feature_columns if col in self.data.columns]
        
        # Prepare features
        X = self.data[available_features].fillna(0)
        
        # Create target variables
        y_top5 = (self.data['finishing_position'] <= 5).astype(int)
        y_podium = (self.data['finishing_position'] <= 3).astype(int)
        y_points = (self.data['finishing_position'] <= 10).astype(int)
        
        # Use top5 as primary target
        y = y_top5
        
        print(f"✅ Training data prepared: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"   Target distribution: {y.value_counts().to_dict()}")
        
        return X.values, y.values, available_features
    
    def train_ensemble_models(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> Dict:
        """
        Train ensemble of models for robust predictions
        """
        print("🤖 Training ensemble models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        self.scalers['standard'] = scaler
        
        # Train multiple models
        models = {}
        
        # 1. Random Forest
        print("  Training Random Forest...")
        rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train, y_train)
        models['random_forest'] = rf_model
        
        # 2. XGBoost
        print("  Training XGBoost...")
        xgb_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss'
        )
        xgb_model.fit(X_train, y_train)
        models['xgboost'] = xgb_model
        
        # 3. LightGBM
        print("  Training LightGBM...")
        lgb_model = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1
        )
        lgb_model.fit(X_train, y_train)
        models['lightgbm'] = lgb_model
        
        # 4. CatBoost
        print("  Training CatBoost...")
        cb_model = cb.CatBoostClassifier(
            iterations=200,
            depth=6,
            learning_rate=0.1,
            random_seed=42,
            verbose=False
        )
        cb_model.fit(X_train, y_train)
        models['catboost'] = cb_model
        
        # 5. Neural Network
        print("  Training Neural Network...")
        nn_model = self._build_neural_network(X_train_scaled.shape[1])
        nn_model.fit(
            X_train_scaled, y_train,
            epochs=100,
            batch_size=32,
            validation_split=0.2,
            callbacks=[
                EarlyStopping(patience=10, restore_best_weights=True),
                ReduceLROnPlateau(patience=5, factor=0.5)
            ],
            verbose=0
        )
        models['neural_network'] = nn_model
        
        # 6. Support Vector Machine
        print("  Training Support Vector Machine...")
        svm_model = SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            probability=True,
            random_state=42
        )
        svm_model.fit(X_train_scaled, y_train)
        models['svm'] = svm_model
        
        # 7. Logistic Regression
        print("  Training Logistic Regression...")
        lr_model = LogisticRegression(
            C=1.0,
            random_state=42,
            max_iter=1000
        )
        lr_model.fit(X_train_scaled, y_train)
        models['logistic_regression'] = lr_model
        
        # Evaluate models
        print("\n📊 Evaluating models...")
        model_scores = {}
        
        for name, model in models.items():
            if name == 'neural_network':
                y_pred = (model.predict(X_test_scaled) > 0.5).astype(int)
            else:
                y_pred = model.predict(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            model_scores[name] = accuracy
            print(f"  {name}: {accuracy:.4f}")
        
        # Create ensemble model
        print("\n🎯 Creating ensemble model...")
        ensemble_models = [
            ('rf', models['random_forest']),
            ('xgb', models['xgboost']),
            ('lgb', models['lightgbm']),
            ('cb', models['catboost']),
            ('svm', models['svm']),
            ('lr', models['logistic_regression'])
        ]
        
        ensemble_model = VotingClassifier(
            estimators=ensemble_models,
            voting='soft'
        )
        ensemble_model.fit(X_train, y_train)
        models['ensemble'] = ensemble_model
        
        # Evaluate ensemble
        y_pred_ensemble = ensemble_model.predict(X_test)
        ensemble_accuracy = accuracy_score(y_test, y_pred_ensemble)
        model_scores['ensemble'] = ensemble_accuracy
        print(f"  Ensemble: {ensemble_accuracy:.4f}")
        
        self.models = models
        self.prediction_accuracy = model_scores
        
        # Calculate feature importance
        self._calculate_feature_importance(feature_names)
        
        print("✅ Ensemble models trained successfully")
        return models
    
    def _build_neural_network(self, input_dim: int) -> Sequential:
        """Build neural network architecture"""
        model = Sequential([
            Dense(128, activation='relu', input_shape=(input_dim,)),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dropout(0.1),
            
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _calculate_feature_importance(self, feature_names: List[str]):
        """Calculate feature importance from different models"""
        print("🔍 Calculating feature importance...")
        
        importance_scores = {}
        
        # Random Forest feature importance
        if 'random_forest' in self.models:
            rf_importance = self.models['random_forest'].feature_importances_
            importance_scores['random_forest'] = dict(zip(feature_names, rf_importance))
        
        # XGBoost feature importance
        if 'xgboost' in self.models:
            xgb_importance = self.models['xgboost'].feature_importances_
            importance_scores['xgboost'] = dict(zip(feature_names, xgb_importance))
        
        # LightGBM feature importance
        if 'lightgbm' in self.models:
            lgb_importance = self.models['lightgbm'].feature_importances_
            importance_scores['lightgbm'] = dict(zip(feature_names, lgb_importance))
        
        # CatBoost feature importance
        if 'catboost' in self.models:
            cb_importance = self.models['catboost'].feature_importances_
            importance_scores['catboost'] = dict(zip(feature_names, cb_importance))
        
        # Calculate average importance
        if importance_scores:
            avg_importance = {}
            for feature in feature_names:
                scores = [scores.get(feature, 0) for scores in importance_scores.values()]
                avg_importance[feature] = np.mean(scores)
            
            importance_scores['average'] = avg_importance
        
        self.feature_importance = importance_scores
        
        # Print top features
        if 'average' in importance_scores:
            sorted_features = sorted(importance_scores['average'].items(), 
                                   key=lambda x: x[1], reverse=True)
            print("📈 Top 10 Most Important Features:")
            for i, (feature, importance) in enumerate(sorted_features[:10]):
                print(f"  {i+1:2d}. {feature}: {importance:.4f}")
    
    def predict_future_race(self, race_data: pd.DataFrame) -> pd.DataFrame:
        """
        Predict future race outcomes
        """
        print("🔮 Predicting future race outcomes...")
        
        if not self.models:
            print("❌ Models not trained. Please train models first.")
            return pd.DataFrame()
        
        # Prepare prediction data
        feature_columns = [
            'driver_avg_finish', 'driver_consistency', 'driver_podium_rate', 'driver_top5_rate',
            'driver_recent_form', 'driver_trend_slope', 'team_avg_finish', 'team_consistency',
            'team_podium_rate', 'team_top5_rate', 'team_development_trend', 'team_reliability',
            'circuit_overtaking', 'circuit_safety_car_prob', 'circuit_grid_importance',
            'circuit_qualifying_correlation'
        ]
        
        # Filter available features
        available_features = [col for col in feature_columns if col in race_data.columns]
        X_pred = race_data[available_features].fillna(0)
        
        # Make predictions with all models
        predictions = {}
        
        for name, model in self.models.items():
            if name == 'neural_network':
                X_pred_scaled = self.scalers['standard'].transform(X_pred)
                pred_proba = model.predict(X_pred_scaled).flatten()
            else:
                pred_proba = model.predict_proba(X_pred)[:, 1]
            
            predictions[f'{name}_top5_prob'] = pred_proba
        
        # Calculate ensemble prediction
        ensemble_probs = []
        for name in ['random_forest', 'xgboost', 'lightgbm', 'catboost', 'svm', 'logistic_regression']:
            if f'{name}_top5_prob' in predictions:
                ensemble_probs.append(predictions[f'{name}_top5_prob'])
        
        if ensemble_probs:
            predictions['ensemble_top5_prob'] = np.mean(ensemble_probs, axis=0)
        
        # Create prediction results
        results = race_data[['driver_name', 'team_name', 'circuit']].copy()
        
        for name, prob in predictions.items():
            results[name] = prob
        
        # Sort by ensemble probability
        if 'ensemble_top5_prob' in results.columns:
            results = results.sort_values('ensemble_top5_prob', ascending=False)
        
        print(f"✅ Predictions generated for {len(results)} drivers")
        return results
    
    def validate_prediction_accuracy(self, test_data: pd.DataFrame) -> Dict:
        """
        Validate prediction accuracy on test data
        """
        print("📊 Validating prediction accuracy...")
        
        if not self.models:
            print("❌ Models not trained. Please train models first.")
            return {}
        
        # Prepare test data
        feature_columns = [
            'driver_avg_finish', 'driver_consistency', 'driver_podium_rate', 'driver_top5_rate',
            'driver_recent_form', 'driver_trend_slope', 'team_avg_finish', 'team_consistency',
            'team_podium_rate', 'team_top5_rate', 'team_development_trend', 'team_reliability',
            'circuit_overtaking', 'circuit_safety_car_prob', 'circuit_grid_importance',
            'circuit_qualifying_correlation'
        ]
        
        available_features = [col for col in feature_columns if col in test_data.columns]
        X_test = test_data[available_features].fillna(0)
        y_test = (test_data['finishing_position'] <= 5).astype(int)
        
        # Make predictions
        predictions = {}
        
        for name, model in self.models.items():
            if name == 'neural_network':
                X_test_scaled = self.scalers['standard'].transform(X_test)
                y_pred = (model.predict(X_test_scaled) > 0.5).astype(int)
            else:
                y_pred = model.predict(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            predictions[name] = accuracy
        
        # Calculate ensemble accuracy
        ensemble_probs = []
        for name in ['random_forest', 'xgboost', 'lightgbm', 'catboost', 'svm', 'logistic_regression']:
            if name in self.models:
                if name == 'neural_network':
                    X_test_scaled = self.scalers['standard'].transform(X_test)
                    prob = self.models[name].predict(X_test_scaled).flatten()
                else:
                    prob = self.models[name].predict_proba(X_test)[:, 1]
                ensemble_probs.append(prob)
        
        if ensemble_probs:
            ensemble_pred = (np.mean(ensemble_probs, axis=0) > 0.5).astype(int)
            ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
            predictions['ensemble'] = ensemble_accuracy
        
        # Print results
        print("📈 Validation Results:")
        for name, accuracy in predictions.items():
            print(f"  {name}: {accuracy:.4f}")
        
        return predictions
    
    def save_models(self, output_dir: str = 'model/advanced'):
        """Save trained models and artifacts"""
        print("💾 Saving models and artifacts...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save models
        for name, model in self.models.items():
            if name == 'neural_network':
                model.save(f'{output_dir}/{name}.keras')
            else:
                joblib.dump(model, f'{output_dir}/{name}.pkl')
        
        # Save scalers
        for name, scaler in self.scalers.items():
            joblib.dump(scaler, f'{output_dir}/scaler_{name}.pkl')
        
        # Save feature importance
        joblib.dump(self.feature_importance, f'{output_dir}/feature_importance.pkl')
        
        # Save prediction accuracy
        joblib.dump(self.prediction_accuracy, f'{output_dir}/prediction_accuracy.pkl')
        
        print(f"✅ Models saved to {output_dir}")
    
    def load_models(self, model_dir: str = 'model/advanced'):
        """Load trained models and artifacts"""
        print("📂 Loading models and artifacts...")
        
        try:
            # Load models
            model_files = {
                'random_forest': 'random_forest.pkl',
                'xgboost': 'xgboost.pkl',
                'lightgbm': 'lightgbm.pkl',
                'catboost': 'catboost.pkl',
                'neural_network': 'neural_network.keras',
                'svm': 'svm.pkl',
                'logistic_regression': 'logistic_regression.pkl',
                'ensemble': 'ensemble.pkl'
            }
            
            for name, filename in model_files.items():
                filepath = f'{model_dir}/{filename}'
                if os.path.exists(filepath):
                    if name == 'neural_network':
                        self.models[name] = load_model(filepath)
                    else:
                        self.models[name] = joblib.load(filepath)
            
            # Load scalers
            scaler_files = ['scaler_standard.pkl']
            for filename in scaler_files:
                filepath = f'{model_dir}/{filename}'
                if os.path.exists(filepath):
                    name = filename.replace('scaler_', '').replace('.pkl', '')
                    self.scalers[name] = joblib.load(filepath)
            
            # Load feature importance
            importance_file = f'{model_dir}/feature_importance.pkl'
            if os.path.exists(importance_file):
                self.feature_importance = joblib.load(importance_file)
            
            # Load prediction accuracy
            accuracy_file = f'{model_dir}/prediction_accuracy.pkl'
            if os.path.exists(accuracy_file):
                self.prediction_accuracy = joblib.load(accuracy_file)
            
            print("✅ Models loaded successfully")
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
    
    def generate_prediction_report(self, predictions: pd.DataFrame) -> str:
        """Generate detailed prediction report"""
        report = []
        report.append("🏁 F1 Race Prediction Report")
        report.append("=" * 50)
        
        if 'ensemble_top5_prob' in predictions.columns:
            report.append("\n📊 Top 5 Probability Rankings:")
            report.append("-" * 30)
            
            for i, (_, row) in enumerate(predictions.head(10).iterrows()):
                driver = row['driver_name']
                team = row['team_name']
                prob = row['ensemble_top5_prob']
                report.append(f"{i+1:2d}. {driver} ({team}): {prob:.3f}")
        
        report.append("\n🎯 Model Confidence:")
        report.append("-" * 20)
        
        if self.prediction_accuracy:
            for name, accuracy in self.prediction_accuracy.items():
                report.append(f"{name}: {accuracy:.3f}")
        
        report.append("\n📈 Key Insights:")
        report.append("-" * 15)
        
        if 'ensemble_top5_prob' in predictions.columns:
            top_prob = predictions['ensemble_top5_prob'].max()
            avg_prob = predictions['ensemble_top5_prob'].mean()
            report.append(f"• Highest probability: {top_prob:.3f}")
            report.append(f"• Average probability: {avg_prob:.3f}")
            report.append(f"• Drivers with >50% chance: {(predictions['ensemble_top5_prob'] > 0.5).sum()}")
        
        return "\n".join(report)

def main():
    """Main function to run the advanced prediction system"""
    print("🎯 Starting Advanced F1 Prediction System")
    print("=" * 50)
    
    # Initialize prediction system
    system = AdvancedF1PredictionSystem()
    
    # Load data
    data = system.load_data()
    if data.empty:
        print("❌ No data available for prediction system")
        return
    
    # Prepare training data
    X, y, feature_names = system.prepare_training_data()
    if X.size == 0:
        print("❌ No training data available")
        return
    
    # Train models
    models = system.train_ensemble_models(X, y, feature_names)
    
    # Save models
    system.save_models()
    
    # Validate accuracy
    validation_results = system.validate_prediction_accuracy(data)
    
    print("\n✅ Advanced F1 Prediction System Complete!")
    print("=" * 50)
    print("System ready for future race predictions!")
    print("\nTo make predictions:")
    print("1. Prepare race data with required features")
    print("2. Call system.predict_future_race(race_data)")
    print("3. Review prediction report")

if __name__ == "__main__":
    main()
