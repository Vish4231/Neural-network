#!/usr/bin/env python3
"""
F1 Prediction Utilities
Contains prediction engine, result formatting, and file management utilities
"""

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import glob

class PredictionEngine:
    """Core prediction engine for F1 races"""
    
    def __init__(self):
        self.model_cache = {}
        print("🔮 Prediction Engine Initialized")
    
    def predict_race(self, lineup: pd.DataFrame, circuit_data: Dict, year: int) -> pd.DataFrame:
        """Generate race predictions based on lineup and circuit data"""
        
        predictions = []
        
        for _, row in lineup.iterrows():
            driver = row['driver_name']
            team = row['team_name']
            
            # Calculate prediction scores
            driver_score = self._calculate_driver_score(driver, circuit_data)
            team_score = self._calculate_team_score(team, circuit_data)
            circuit_adaptation = self._calculate_circuit_adaptation(driver, team, circuit_data)
            form_factor = self._get_recent_form(driver)
            
            # Composite prediction score (form matters more for current predictions)
            composite_score = (
                driver_score * 0.25 +        # Base ability (reduced)
                team_score * 0.35 +          # Team performance (increased - car matters most)
                circuit_adaptation * 0.15 +   # Circuit fit (reduced)
                form_factor * 0.25            # Recent form (increased - current performance matters)
            )
            
            # Add randomness for realism
            random_factor = np.random.normal(1.0, 0.05)
            final_score = composite_score * random_factor
            
            # Convert to probabilities
            win_prob = min(0.85, max(0.01, final_score * 0.7))
            podium_prob = min(0.90, max(0.05, final_score * 0.8))
            top5_prob = min(0.95, max(0.10, final_score * 0.9))
            points_prob = min(0.98, max(0.20, final_score * 1.1))
            
            predictions.append({
                'position_prediction': 0,  # Will be set after sorting
                'driver_name': driver,
                'team_name': team,
                'performance_score': final_score,
                'win_probability': win_prob,
                'podium_probability': podium_prob,
                'top5_probability': top5_prob,
                'points_probability': points_prob,
                'driver_rating': driver_score,
                'team_rating': team_score,
                'circuit_adaptation': circuit_adaptation,
                'recent_form': form_factor
            })
        
        # Create DataFrame and sort by performance score
        df = pd.DataFrame(predictions)
        df = df.sort_values('performance_score', ascending=False).reset_index(drop=True)
        
        # Assign predicted positions
        df['position_prediction'] = range(1, len(df) + 1)
        
        return df
    
    def _calculate_driver_score(self, driver: str, circuit_data: Dict) -> float:
        """Calculate driver-specific performance score"""
        
        # Base driver ratings (realistic assessment based on career performance)
        driver_ratings = {
            # Elite tier
            'Max Verstappen': 0.98,          # Reigning champion, peak performance
            'Lewis Hamilton': 0.96,          # 7x champion, still elite
            'Fernando Alonso': 0.94,         # Veteran brilliance
            
            # Top tier current performers  
            'Charles Leclerc': 0.93,         # Ferrari ace, race winner
            'Lando Norris': 0.91,            # Proven race winner
            'George Russell': 0.89,          # Consistent, race winner
            'Oscar Piastri': 0.87,           # Rising star, multiple wins
            'Carlos Sainz': 0.86,            # Experienced race winner
            
            # Solid performers
            'Pierre Gasly': 0.82,            # Race winner, experienced
            'Alexander Albon': 0.80,         # Strong comeback story
            'Esteban Ocon': 0.79,            # Race winner, consistent
            'Nico Hulkenberg': 0.78,        # Veteran, no wins but fast
            
            # Inconsistent/developing
            'Yuki Tsunoda': 0.74,            # Pace but inconsistent
            'Liam Lawson': 0.72,             # Promising but limited data
            'Lance Stroll': 0.69,            # Adequate but not elite
            
            # Rookies/New faces (potential-based)
            'Andrea Kimi Antonelli': 0.78,   # Highly rated rookie
            'Oliver Bearman': 0.75,          # F1 experience helps
            'Franco Colapinto': 0.73,        # Showed promise at Williams
            'Isack Hadjar': 0.71,            # F2 talent
            'Gabriel Bortoleto': 0.70        # F2 champion but rookie
        }
        
        base_rating = driver_ratings.get(driver, 0.60)
        
        # Circuit-specific adjustments
        circuit_bonus = 0
        overtaking_difficulty = circuit_data.get('overtaking_difficulty', 5)
        
        # Skilled drivers benefit more on difficult circuits
        if driver in ['Max Verstappen', 'Lewis Hamilton', 'Fernando Alonso']:
            circuit_bonus = (overtaking_difficulty / 10) * 0.1
        
        return min(1.0, base_rating + circuit_bonus)
    
    def _calculate_team_score(self, team: str, circuit_data: Dict) -> float:
        """Calculate team-specific performance score"""
        
        team_ratings = {
            # Top 3 (championship contenders)
            'Red Bull Racing': 0.94,        # Still strong but competition closer
            'Ferrari': 0.93,                # Hamilton boost + continued development
            'McLaren': 0.92,                # 2024 momentum continues
            
            # Strong midfield
            'Mercedes': 0.88,               # Slight decline without Hamilton
            'Aston Martin': 0.81,           # Consistent midfield
            
            # Midfield pack
            'Alpine': 0.76,                 # Steady midfield team
            'Williams': 0.78,               # Sainz upgrade boosts performance
            'Racing Bulls': 0.72,           # Sister team, decent package
            'Haas': 0.74,                   # Ocon brings experience
            'Sauber': 0.66                  # Struggling at back
        }
        
        base_rating = team_ratings.get(team, 0.60)
        
        # Circuit-specific team adjustments
        downforce_setup = circuit_data.get('downforce_setup', 0.5)
        
        # Some teams excel at high/low downforce circuits
        circuit_bonus = 0
        if team == 'Red Bull Racing':
            circuit_bonus = 0.02  # Strong everywhere
        elif team == 'Ferrari' and downforce_setup > 0.7:
            circuit_bonus = 0.03  # Strong at high downforce circuits
        elif team == 'Mercedes' and circuit_data.get('tire_degradation') == 'high':
            circuit_bonus = 0.02  # Good tire management
        
        return min(1.0, base_rating + circuit_bonus)
    
    def _calculate_circuit_adaptation(self, driver: str, team: str, circuit_data: Dict) -> float:
        """Calculate how well driver/team adapts to specific circuit"""
        
        # Base adaptation score
        adaptation = 0.7
        
        # Circuit characteristics
        overtaking_difficulty = circuit_data.get('overtaking_difficulty', 5)
        safety_car_prob = circuit_data.get('safety_car_prob', 0.3)
        weather_variability = circuit_data.get('weather_variability', 0.5)
        
        # Driver-specific adaptations
        if driver in ['Lewis Hamilton', 'Fernando Alonso']:
            # Veterans excel in challenging conditions
            adaptation += weather_variability * 0.2
            
        if driver in ['Max Verstappen', 'Charles Leclerc']:
            # Aggressive drivers thrive with safety cars
            adaptation += safety_car_prob * 0.1
        
        # Team-specific adaptations
        if team == 'Red Bull Racing':
            adaptation += 0.1  # Excellent adaptability
        elif team in ['Mercedes', 'Ferrari']:
            adaptation += 0.05  # Good adaptability
        
        return min(1.0, max(0.3, adaptation))
    
    def _get_recent_form(self, driver: str) -> float:
        """Get recent form factor based on actual 2024 season performance"""
        
        # Recent form based on 2024 season performance and trajectory
        recent_form = {
            # Championship contenders (excellent form)
            'Max Verstappen': 0.98,          # Dominant 2024 champion
            'Lando Norris': 0.96,            # Multiple wins in 2024
            'Charles Leclerc': 0.94,         # Strong 2024 with Ferrari
            'Oscar Piastri': 0.93,           # Breakthrough 2024 season
            
            # Strong performers
            'Lewis Hamilton': 0.88,          # Solid 2024, Ferrari move excitement
            'George Russell': 0.86,          # Consistent Mercedes performer
            'Carlos Sainz': 0.85,            # Strong final Ferrari season
            'Fernando Alonso': 0.82,         # Veteran consistency
            
            # Solid midfield
            'Pierre Gasly': 0.78,            # Decent Alpine season
            'Alexander Albon': 0.76,         # Good Williams performances
            'Nico Hulkenberg': 0.74,        # Veteran reliability
            'Esteban Ocon': 0.73,            # Consistent but unspectacular
            
            # Inconsistent/struggling
            'Yuki Tsunoda': 0.65,            # Inconsistent 2024, RB promotion questionable
            'Lance Stroll': 0.58,            # Below par 2024
            'Liam Lawson': 0.60,             # Limited opportunities
            
            # Rookies/New drivers (based on junior series)
            'Andrea Kimi Antonelli': 0.85,   # F2 champion, high potential
            'Oliver Bearman': 0.82,          # Strong F2 season, F1 experience
            'Franco Colapinto': 0.75,        # Showed well in Williams stint
            'Isack Hadjar': 0.70,            # F2 talent
            'Gabriel Bortoleto': 0.72        # F2 champion
        }
        
        return recent_form.get(driver, 0.60)

class ResultFormatter:
    """Format prediction results for display"""
    
    def __init__(self):
        self.emoji_positions = {
            1: "🥇", 2: "🥈", 3: "🥉", 4: "4️⃣", 5: "5️⃣",
            6: "6️⃣", 7: "7️⃣", 8: "8️⃣", 9: "9️⃣", 10: "🔟"
        }
    
    def format_predictions(self, predictions: pd.DataFrame, circuit_data: Dict, interactive: bool = False) -> str:
        """Format predictions for display"""
        
        circuit_name = circuit_data.get('name', 'Unknown Circuit')
        country = circuit_data.get('country', 'Unknown')
        
        output = []
        
        # Header
        output.append("🏁 " + "="*60)
        output.append(f"🏎️  F1 RACE PREDICTION: {circuit_name}")
        output.append(f"📍 Location: {country}")
        output.append(f"📏 Length: {circuit_data.get('length_km', 0):.3f}km")
        output.append(f"🔄 Turns: {circuit_data.get('turns', 0)}")
        output.append("🏁 " + "="*60)
        
        # Top 10 predictions
        output.append("\n🏆 RACE PREDICTIONS - TOP 10:")
        output.append("-" * 50)
        
        for i, (_, row) in enumerate(predictions.head(10).iterrows()):
            pos = i + 1
            emoji = self.emoji_positions.get(pos, f"{pos:2d}.")
            driver = row['driver_name']
            team = row['team_name']
            score = row['performance_score']
            win_prob = row['win_probability']
            
            output.append(f"{emoji} {driver:<20} ({team:<12}) Score: {score:.3f} | Win: {win_prob:.1%}")
        
        # Probabilities section
        output.append(f"\n🎯 WIN PROBABILITIES:")
        output.append("-" * 30)
        
        win_favorites = predictions.nlargest(5, 'win_probability')
        for _, row in win_favorites.iterrows():
            driver = row['driver_name']
            team = row['team_name']
            prob = row['win_probability']
            output.append(f"• {driver} ({team}): {prob:.1%}")
        
        # Podium predictions
        output.append(f"\n🏆 PODIUM PROBABILITIES:")
        output.append("-" * 30)
        
        podium_favorites = predictions.nlargest(5, 'podium_probability')
        for _, row in podium_favorites.iterrows():
            driver = row['driver_name']
            team = row['team_name']
            prob = row['podium_probability']
            output.append(f"• {driver} ({team}): {prob:.1%}")
        
        # Circuit insights
        if interactive:
            output.append(self._format_circuit_insights(circuit_data))
            output.append(self._format_team_analysis(predictions))
        
        # Footer
        output.append(f"\n🏁 " + "="*60)
        output.append(f"⚡ Prediction generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return "\n".join(output)
    
    def _format_circuit_insights(self, circuit_data: Dict) -> str:
        """Format circuit-specific insights"""
        
        insights = ["\n🛣️  CIRCUIT INSIGHTS:"]
        insights.append("-" * 25)
        
        # Overtaking difficulty
        difficulty = circuit_data.get('overtaking_difficulty', 5)
        if difficulty >= 8:
            insights.append("• Overtaking: Very Difficult - Track position crucial")
        elif difficulty >= 6:
            insights.append("• Overtaking: Moderate - Some opportunities available")
        else:
            insights.append("• Overtaking: Relatively Easy - Multiple opportunities")
        
        # Safety car probability
        sc_prob = circuit_data.get('safety_car_prob', 0.3)
        if sc_prob >= 0.6:
            insights.append(f"• Safety Car: High probability ({sc_prob:.0%}) - Strategy crucial")
        elif sc_prob >= 0.3:
            insights.append(f"• Safety Car: Moderate probability ({sc_prob:.0%})")
        else:
            insights.append(f"• Safety Car: Low probability ({sc_prob:.0%})")
        
        # Weather variability
        weather = circuit_data.get('weather_variability', 0.5)
        if weather >= 0.7:
            insights.append("• Weather: Highly variable - Expect surprises")
        elif weather >= 0.4:
            insights.append("• Weather: Somewhat variable")
        else:
            insights.append("• Weather: Stable conditions expected")
        
        # Tire degradation
        tire_deg = circuit_data.get('tire_degradation', 'medium')
        insights.append(f"• Tire Degradation: {tire_deg.title()}")
        
        return "\n".join(insights)
    
    def _format_team_analysis(self, predictions: pd.DataFrame) -> str:
        """Format team-by-team analysis"""
        
        analysis = ["\n🏎️  TEAM ANALYSIS:"]
        analysis.append("-" * 20)
        
        team_avg = predictions.groupby('team_name').agg({
            'performance_score': 'mean',
            'win_probability': 'max'
        }).sort_values('performance_score', ascending=False)
        
        for team, row in team_avg.head(5).iterrows():
            score = row['performance_score']
            win_prob = row['win_probability']
            analysis.append(f"• {team:<15}: Score {score:.3f} | Best Win Chance: {win_prob:.1%}")
        
        return "\n".join(analysis)

class FileManager:
    """Manage prediction files and output"""
    
    def __init__(self):
        self.output_dir = "predictions"
        self.ensure_output_dir()
    
    def ensure_output_dir(self):
        """Ensure output directory exists"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"📁 Created predictions directory: {self.output_dir}")
    
    def save_predictions(self, predictions: pd.DataFrame, circuit: str, year: int) -> str:
        """Save predictions to CSV file"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{circuit}_{year}_{timestamp}.csv"
        filepath = os.path.join(self.output_dir, filename)
        
        # Add metadata columns
        save_df = predictions.copy()
        save_df['circuit'] = circuit
        save_df['year'] = year
        save_df['prediction_date'] = datetime.now().isoformat()
        
        # Reorder columns for better readability
        columns = [
            'position_prediction', 'driver_name', 'team_name',
            'performance_score', 'win_probability', 'podium_probability',
            'top5_probability', 'points_probability',
            'circuit', 'year', 'prediction_date'
        ]
        
        # Include additional columns that exist
        for col in save_df.columns:
            if col not in columns:
                columns.append(col)
        
        save_df = save_df[columns]
        save_df.to_csv(filepath, index=False, float_format='%.4f')
        
        return filename
    
    def get_recent_predictions(self, limit: int = 10) -> List[Dict]:
        """Get list of recent prediction files"""
        
        if not os.path.exists(self.output_dir):
            return []
        
        # Get all CSV files in predictions directory
        pattern = os.path.join(self.output_dir, "*.csv")
        files = glob.glob(pattern)
        
        # Sort by modification time (newest first)
        files.sort(key=os.path.getmtime, reverse=True)
        
        predictions = []
        for filepath in files[:limit]:
            try:
                stat = os.stat(filepath)
                predictions.append({
                    'filename': os.path.basename(filepath),
                    'path': filepath,
                    'date': datetime.fromtimestamp(stat.st_mtime),
                    'size_kb': round(stat.st_size / 1024, 1)
                })
            except OSError:
                continue
        
        return predictions
    
    def load_prediction(self, filename: str) -> Optional[pd.DataFrame]:
        """Load a prediction file"""
        
        filepath = os.path.join(self.output_dir, filename)
        
        if not os.path.exists(filepath):
            return None
        
        try:
            return pd.read_csv(filepath)
        except Exception as e:
            print(f"❌ Error loading {filename}: {e}")
            return None
    
    def export_predictions(self, predictions: pd.DataFrame, format: str, circuit: str, year: int) -> str:
        """Export predictions in different formats"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format == 'json':
            filename = f"{circuit}_{year}_{timestamp}.json"
            filepath = os.path.join(self.output_dir, filename)
            predictions.to_json(filepath, orient='records', indent=2)
        
        elif format == 'html':
            filename = f"{circuit}_{year}_{timestamp}.html"
            filepath = os.path.join(self.output_dir, filename)
            html_content = self._generate_html_report(predictions, circuit, year)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(html_content)
        
        else:  # Default to CSV
            return self.save_predictions(predictions, circuit, year)
        
        return filename
    
    def _generate_html_report(self, predictions: pd.DataFrame, circuit: str, year: int) -> str:
        """Generate HTML report for predictions"""
        
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>F1 Predictions: {circuit} {year}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #e10600; text-align: center; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #f8f9fa; font-weight: bold; }}
        tr:hover {{ background-color: #f5f5f5; }}
        .position {{ font-weight: bold; color: #e10600; }}
        .probability {{ color: #28a745; }}
        .footer {{ text-align: center; color: #666; margin-top: 30px; font-size: 14px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🏎️ F1 Race Predictions</h1>
        <h2>{circuit} {year}</h2>
        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <table>
            <thead>
                <tr>
                    <th>Position</th>
                    <th>Driver</th>
                    <th>Team</th>
                    <th>Performance Score</th>
                    <th>Win Probability</th>
                    <th>Podium Probability</th>
                    <th>Top 5 Probability</th>
                </tr>
            </thead>
            <tbody>
        """
        
        for _, row in predictions.iterrows():
            html += f"""
                <tr>
                    <td class="position">{row['position_prediction']}</td>
                    <td>{row['driver_name']}</td>
                    <td>{row['team_name']}</td>
                    <td>{row['performance_score']:.3f}</td>
                    <td class="probability">{row['win_probability']:.1%}</td>
                    <td class="probability">{row['podium_probability']:.1%}</td>
                    <td class="probability">{row['top5_probability']:.1%}</td>
                </tr>
            """
        
        html += """
            </tbody>
        </table>
        
        <div class="footer">
            <p>🏁 Generated by F1 Race Predictor</p>
        </div>
    </div>
</body>
</html>
        """
        
        return html

# Utility functions for quick operations
def quick_predict(circuit: str, year: int = 2025) -> pd.DataFrame:
    """Quick prediction function for scripting"""
    from f1_config import CircuitDatabase, DriverDatabase
    
    circuit_db = CircuitDatabase()
    driver_db = DriverDatabase()
    engine = PredictionEngine()
    
    circuit_key = circuit_db.find_circuit(circuit)
    if not circuit_key:
        print(f"❌ Circuit '{circuit}' not found")
        return pd.DataFrame()
    
    lineup = driver_db.get_2025_lineup() if year >= 2025 else driver_db.get_historical_lineup(year)
    if lineup is None or lineup.empty:
        print(f"❌ No lineup data for {year}")
        return pd.DataFrame()
    
    circuit_data = circuit_db.get_circuit_data(circuit_key)
    return engine.predict_race(lineup, circuit_data, year)

def format_quick_results(predictions: pd.DataFrame, circuit: str) -> str:
    """Quick format for simple results"""
    if predictions.empty:
        return "No predictions available"
    
    output = [f"\n🏁 {circuit} - Top 5 Predictions:"]
    output.append("-" * 30)
    
    for i, (_, row) in enumerate(predictions.head(5).iterrows()):
        pos = i + 1
        driver = row['driver_name']
        team = row['team_name']
        win_prob = row['win_probability']
        
        output.append(f"{pos}. {driver} ({team}) - {win_prob:.1%}")
    
    return "\n".join(output)