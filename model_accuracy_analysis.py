#!/usr/bin/env python3
"""
F1 Prediction Model Accuracy Analysis and Improvement Recommendations
Comparing predicted vs actual Spa 2025 results
"""

import pandas as pd
import numpy as np

def analyze_prediction_accuracy():
    """Analyze the accuracy of Spa 2025 predictions vs actual results"""
    
    # Load predicted results
    predicted = pd.read_csv('predictions_future_race.csv')
    
    # Load actual results
    actual = pd.read_csv('spa_2025_actual_results.csv')
    
    print("=== F1 PREDICTION MODEL ACCURACY ANALYSIS ===\n")
    
    # Normalize driver names for comparison
    def normalize_name(name):
        name = str(name).strip()
        # Handle common name variations
        name_map = {
            'O. Piastri': 'Oscar Piastri',
            'L. Norris': 'Lando Norris', 
            'C. Leclerc': 'Charles Leclerc',
            'M. Verstappen': 'Max Verstappen',
            'G. Russell': 'George Russell',
            'A. Albon': 'Alexander Albon',
            'L. Hamilton': 'Lewis Hamilton',
            'L. Lawson': 'Liam Lawson',
            'G. Bortoleto': 'Gabriel Bortoleto',
            'P. Gasly': 'Pierre Gasly',
            'O. Bearman': 'Oliver Bearman',
            'N. Hülkenberg': 'Nico Hulkenberg',
            'Y. Tsunoda': 'Yuki Tsunoda',
            'L. Stroll': 'Lance Stroll',
            'E. Ocon': 'Esteban Ocon',
            'A.K. Antonelli': 'Kimi Antonelli',
            'F. Alonso': 'Fernando Alonso',
            'C. Sainz Jr.': 'Carlos Sainz',
            'F. Colapinto': 'Franco Colapinto',
            'I. Hadjar': 'Isack Hadjar'
        }
        return name_map.get(name, name)
    
    actual['Driver_normalized'] = actual['Driver'].apply(normalize_name)
    
    # Create comparison dataframe
    comparison = []
    for _, actual_row in actual.iterrows():
        driver = actual_row['Driver_normalized']
        actual_pos = actual_row['Position']
        
        # Find predicted position
        pred_row = predicted[predicted['driver_name'] == driver]
        if not pred_row.empty:
            pred_pos = pred_row['race_rank'].iloc[0]
            prob_top5 = pred_row['race_prob_top5'].iloc[0]
        else:
            pred_pos = None
            prob_top5 = None
            
        comparison.append({
            'driver': driver,
            'actual_position': actual_pos,
            'predicted_position': pred_pos,
            'position_error': abs(actual_pos - pred_pos) if pred_pos else None,
            'prob_top5': prob_top5,
            'actual_top5': actual_pos <= 5,
            'predicted_top5': prob_top5 > 0.5 if prob_top5 else None
        })
    
    comp_df = pd.DataFrame(comparison)
    
    print("POSITION PREDICTION ACCURACY:")
    print("=" * 50)
    valid_predictions = comp_df.dropna(subset=['position_error'])
    if not valid_predictions.empty:
        mean_error = valid_predictions['position_error'].mean()
        median_error = valid_predictions['position_error'].median()
        max_error = valid_predictions['position_error'].max()
        
        print(f"Mean Position Error: {mean_error:.2f} positions")
        print(f"Median Position Error: {median_error:.2f} positions") 
        print(f"Maximum Position Error: {max_error:.0f} positions")
        
        # Accuracy within different ranges
        within_3 = (valid_predictions['position_error'] <= 3).mean() * 100
        within_5 = (valid_predictions['position_error'] <= 5).mean() * 100
        within_10 = (valid_predictions['position_error'] <= 10).mean() * 100
        
        print(f"Predictions within 3 positions: {within_3:.1f}%")
        print(f"Predictions within 5 positions: {within_5:.1f}%")
        print(f"Predictions within 10 positions: {within_10:.1f}%")
    
    print("\nTOP 5 PREDICTION ACCURACY:")
    print("=" * 50)
    top5_predictions = comp_df.dropna(subset=['predicted_top5'])
    if not top5_predictions.empty:
        top5_accuracy = (top5_predictions['actual_top5'] == top5_predictions['predicted_top5']).mean() * 100
        print(f"Top 5 Classification Accuracy: {top5_accuracy:.1f}%")
        
        # Precision and Recall for Top 5
        true_positives = ((top5_predictions['actual_top5'] == True) & (top5_predictions['predicted_top5'] == True)).sum()
        false_positives = ((top5_predictions['actual_top5'] == False) & (top5_predictions['predicted_top5'] == True)).sum()
        false_negatives = ((top5_predictions['actual_top5'] == True) & (top5_predictions['predicted_top5'] == False)).sum()
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        
        print(f"Top 5 Precision: {precision:.2f}")
        print(f"Top 5 Recall: {recall:.2f}")
    
    print("\nDETAILED COMPARISON:")
    print("=" * 50)
    print(f"{'Driver':<20} {'Actual':<8} {'Predicted':<10} {'Error':<8} {'Top5 Prob':<10}")
    print("-" * 70)
    
    for _, row in comp_df.iterrows():
        error_str = f"{row['position_error']:.0f}" if pd.notna(row['position_error']) else "N/A"
        prob_str = f"{row['prob_top5']:.3f}" if pd.notna(row['prob_top5']) else "N/A"
        pred_str = f"{row['predicted_position']:.0f}" if pd.notna(row['predicted_position']) else "N/A"
        
        print(f"{row['driver']:<20} {row['actual_position']:<8} {pred_str:<10} {error_str:<8} {prob_str:<10}")
    
    print("\n" + "=" * 70)
    print("KEY ISSUES IDENTIFIED:")
    print("=" * 70)
    
    # Identify major prediction errors
    major_errors = comp_df[comp_df['position_error'] > 10].dropna()
    if not major_errors.empty:
        print("\nMAJOR PREDICTION ERRORS (>10 positions):")
        for _, row in major_errors.iterrows():
            print(f"- {row['driver']}: Predicted {row['predicted_position']:.0f}, Actual {row['actual_position']} (Error: {row['position_error']:.0f})")
    
    # Top 5 misclassifications
    top5_errors = comp_df[(comp_df['actual_top5'] != comp_df['predicted_top5'])].dropna()
    if not top5_errors.empty:
        print(f"\nTOP 5 MISCLASSIFICATIONS:")
        for _, row in top5_errors.iterrows():
            actual_str = "Top 5" if row['actual_top5'] else "Not Top 5"
            pred_str = "Top 5" if row['predicted_top5'] else "Not Top 5"
            print(f"- {row['driver']}: Predicted {pred_str}, Actually {actual_str}")
    
    return comp_df

def generate_improvement_recommendations():
    """Generate specific recommendations for improving model accuracy"""
    
    print("\n" + "=" * 70)
    print("MODEL IMPROVEMENT RECOMMENDATIONS")
    print("=" * 70)
    
    recommendations = [
        {
            "category": "Data Quality & Features",
            "items": [
                "Add more recent 2024-2025 race data for better trend capture",
                "Include qualifying performance as a stronger predictor",
                "Add weather conditions (rain probability was high at Spa)",
                "Include tire strategy and pit stop timing features",
                "Add driver form over last 5 races instead of just 3",
                "Include team development trajectory (car upgrades)",
                "Add circuit-specific driver performance history"
            ]
        },
        {
            "category": "Feature Engineering",
            "items": [
                "Create interaction features between driver skill and track characteristics",
                "Add relative team performance vs field strength",
                "Include championship position pressure effects",
                "Add safety car probability impact on race strategy",
                "Create dynamic form features that weight recent races more heavily",
                "Add head-to-head driver comparison features"
            ]
        },
        {
            "category": "Model Architecture",
            "items": [
                "Implement separate models for different race types (street vs permanent)",
                "Add ensemble methods with circuit-specific model weighting",
                "Use gradient boosting with custom loss functions for position prediction",
                "Implement neural networks with attention mechanisms for driver interactions",
                "Add calibration layers to improve probability estimates",
                "Use multi-task learning for position and points prediction simultaneously"
            ]
        },
        {
            "category": "Training Strategy",
            "items": [
                "Use time-series cross-validation to respect temporal order",
                "Implement class balancing for top 5 vs non-top 5 predictions",
                "Add regularization to prevent overfitting to historical patterns",
                "Use early stopping based on recent season validation",
                "Implement online learning to adapt to 2025 season changes",
                "Add uncertainty quantification to prediction confidence"
            ]
        },
        {
            "category": "Validation & Testing",
            "items": [
                "Create holdout test sets for each circuit type",
                "Implement rolling window validation for temporal data",
                "Add prediction interval estimation",
                "Test model performance across different weather conditions",
                "Validate predictions against betting odds for calibration",
                "Create simulation-based backtesting framework"
            ]
        }
    ]
    
    for rec in recommendations:
        print(f"\n{rec['category'].upper()}:")
        print("-" * len(rec['category']))
        for i, item in enumerate(rec['items'], 1):
            print(f"{i}. {item}")
    
    print(f"\n{'PRIORITY ACTIONS FOR IMMEDIATE IMPROVEMENT:'}")
    print("=" * 50)
    priority_actions = [
        "Update training data with all available 2025 race results",
        "Add qualifying position as primary feature (strongest predictor)",
        "Implement weather-aware predictions (crucial for Spa)",
        "Create circuit-specific models or feature weights",
        "Add recent form weighting (last 3-5 races more important)",
        "Implement proper cross-validation for temporal data"
    ]
    
    for i, action in enumerate(priority_actions, 1):
        print(f"{i}. {action}")

if __name__ == "__main__":
    comp_df = analyze_prediction_accuracy()
    generate_improvement_recommendations()
    
    # Save detailed comparison
    comp_df.to_csv('spa_prediction_analysis.csv', index=False)
    print(f"\nDetailed analysis saved to 'spa_prediction_analysis.csv'")
