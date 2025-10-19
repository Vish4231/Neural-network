#!/usr/bin/env python3
"""
F1 Race Predictor - Universal CLI Interface
Easy-to-use command-line tool for predicting any F1 Grand Prix

Usage:
    python f1_predictor.py                          # Interactive mode
    python f1_predictor.py spa 2025                 # Quick prediction
    python f1_predictor.py --list-circuits          # Show available circuits
    python f1_predictor.py --interactive            # Force interactive mode
    python f1_predictor.py monaco 2025 --save       # Save predictions to file
"""

import argparse
import sys
import os
from datetime import datetime
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple

# Add src to path
sys.path.append('src')

# Import configuration and utilities
from f1_config import F1Config, CircuitDatabase, DriverDatabase
from f1_utils import PredictionEngine, ResultFormatter, FileManager

class F1Predictor:
    def __init__(self):
        self.config = F1Config()
        self.circuit_db = CircuitDatabase()
        self.driver_db = DriverDatabase()
        self.prediction_engine = PredictionEngine()
        self.formatter = ResultFormatter()
        self.file_manager = FileManager()
        
        print("🏁 F1 Race Predictor Initialized")
        
    def interactive_mode(self):
        """Run interactive prediction mode"""
        print("\n" + "="*60)
        print("🏎️  F1 RACE PREDICTOR - INTERACTIVE MODE")
        print("="*60)
        
        while True:
            print("\n📋 Main Menu:")
            print("1. Predict a race")
            print("2. List available circuits")
            print("3. View 2025 driver lineup")
            print("4. View recent predictions")
            print("5. Settings")
            print("6. Exit")
            
            choice = input("\nSelect option (1-6): ").strip()
            
            if choice == '1':
                self._predict_race_interactive()
            elif choice == '2':
                self._list_circuits()
            elif choice == '3':
                self._show_driver_lineup()
            elif choice == '4':
                self._show_recent_predictions()
            elif choice == '5':
                self._settings_menu()
            elif choice == '6':
                print("👋 Thanks for using F1 Race Predictor!")
                break
            else:
                print("❌ Invalid choice. Please try again.")
    
    def _predict_race_interactive(self):
        """Interactive race prediction"""
        print("\n🔮 Race Prediction Wizard")
        print("-" * 30)
        
        # Step 1: Select circuit
        circuit = self._select_circuit_interactive()
        if not circuit:
            return
        
        # Step 2: Select year
        year = self._select_year_interactive()
        if not year:
            return
        
        # Step 3: Confirm or customize lineup
        lineup = self._select_lineup_interactive(year)
        if lineup is None:
            return
        
        # Step 4: Run prediction
        print(f"\n🏁 Predicting {circuit} {year}...")
        self._run_prediction(circuit, year, lineup, save=True, interactive=True)
    
    def _select_circuit_interactive(self) -> Optional[str]:
        """Interactive circuit selection"""
        circuits = self.circuit_db.get_all_circuits()
        
        print(f"\n🏁 Available Circuits ({len(circuits)}):")
        print("-" * 30)
        
        for i, (key, data) in enumerate(circuits.items(), 1):
            name = data['name']
            country = data.get('country', 'Unknown')
            length = data.get('length_km', 0)
            print(f"{i:2d}. {name} ({country}) - {length}km")
        
        while True:
            try:
                choice = input(f"\nSelect circuit (1-{len(circuits)}) or type name: ").strip()
                
                # Try numeric selection
                if choice.isdigit():
                    idx = int(choice) - 1
                    if 0 <= idx < len(circuits):
                        return list(circuits.keys())[idx]
                
                # Try name matching
                circuit_key = self.circuit_db.find_circuit(choice)
                if circuit_key:
                    return circuit_key
                
                print("❌ Invalid selection. Try again or type 'back' to return.")
                if choice.lower() == 'back':
                    return None
                    
            except (ValueError, IndexError):
                print("❌ Invalid input. Please try again.")
    
    def _select_year_interactive(self) -> Optional[int]:
        """Interactive year selection"""
        current_year = datetime.now().year
        
        print(f"\n📅 Select Year:")
        print(f"Current year: {current_year}")
        print("Recommended: 2025 (latest driver lineup)")
        
        while True:
            year_input = input(f"Enter year ({current_year-5}-{current_year+5}) [2025]: ").strip()
            
            if not year_input:
                return 2025
            
            if year_input.lower() == 'back':
                return None
            
            try:
                year = int(year_input)
                if current_year - 5 <= year <= current_year + 5:
                    return year
                else:
                    print(f"❌ Year must be between {current_year-5} and {current_year+5}")
            except ValueError:
                print("❌ Please enter a valid year")
    
    def _select_lineup_interactive(self, year: int) -> Optional[pd.DataFrame]:
        """Interactive lineup selection"""
        print(f"\n👥 Driver Lineup for {year}:")
        print("-" * 30)
        
        if year >= 2025:
            lineup = self.driver_db.get_2025_lineup()
            print("Using 2025 official lineup:")
            self._display_lineup(lineup)
            
            choice = input("\nUse this lineup? (Y/n/customize): ").strip().lower()
            if choice in ['', 'y', 'yes']:
                return lineup
            elif choice in ['n', 'no', 'back']:
                return None
            elif choice in ['c', 'customize', 'custom']:
                return self._customize_lineup(lineup)
        else:
            # For historical years, use historical data if available
            lineup = self.driver_db.get_historical_lineup(year)
            if lineup is not None:
                print(f"Using {year} historical lineup:")
                self._display_lineup(lineup)
                return lineup
            else:
                print(f"❌ No lineup data available for {year}")
                return None
    
    def _display_lineup(self, lineup: pd.DataFrame):
        """Display driver lineup in a nice format"""
        teams = lineup.groupby('team_name')
        for team_name, drivers in teams:
            driver_names = drivers['driver_name'].tolist()
            print(f"  🏎️  {team_name:<15}: {', '.join(driver_names)}")
    
    def _customize_lineup(self, lineup: pd.DataFrame) -> pd.DataFrame:
        """Allow user to customize the lineup"""
        print("\n🛠️  Lineup Customization")
        print("(Feature coming soon - using default lineup for now)")
        return lineup
    
    def _list_circuits(self):
        """List all available circuits"""
        circuits = self.circuit_db.get_all_circuits()
        
        print(f"\n🏁 Available Circuits ({len(circuits)}):")
        print("="*60)
        
        for key, data in circuits.items():
            name = data['name']
            country = data.get('country', 'Unknown')
            length = data.get('length_km', 0)
            turns = data.get('turns', 0)
            difficulty = data.get('overtaking_difficulty', 5)
            
            print(f"🏁 {name}")
            print(f"   Key: {key}")
            print(f"   Location: {country}")
            print(f"   Length: {length}km | Turns: {turns} | Difficulty: {difficulty}/10")
            print()
    
    def _show_driver_lineup(self):
        """Show 2025 driver lineup"""
        lineup = self.driver_db.get_2025_lineup()
        
        print("\n👥 2025 F1 Driver Lineup:")
        print("="*50)
        self._display_lineup(lineup)
        
        print(f"\nTotal drivers: {len(lineup)}")
        print(f"Teams: {lineup['team_name'].nunique()}")
    
    def _show_recent_predictions(self):
        """Show recent prediction files"""
        predictions = self.file_manager.get_recent_predictions(limit=10)
        
        if not predictions:
            print("\n📂 No recent predictions found")
            return
        
        print(f"\n📂 Recent Predictions ({len(predictions)}):")
        print("-" * 40)
        
        for i, pred in enumerate(predictions, 1):
            print(f"{i:2d}. {pred['filename']}")
            print(f"    Date: {pred['date'].strftime('%Y-%m-%d %H:%M')}")
            print(f"    Size: {pred['size_kb']} KB")
            print()
        
        choice = input("View a prediction file? (enter number or press Enter to continue): ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(predictions):
            self._view_prediction_file(predictions[int(choice)-1]['path'])
    
    def _view_prediction_file(self, filepath: str):
        """View a prediction file"""
        try:
            df = pd.read_csv(filepath)
            print(f"\n📊 Prediction Results ({os.path.basename(filepath)}):")
            print("-" * 50)
            print(df.head(10).to_string(index=False))
        except Exception as e:
            print(f"❌ Error reading file: {e}")
    
    def _settings_menu(self):
        """Settings and configuration menu"""
        print("\n⚙️  Settings")
        print("-" * 20)
        print("1. Output format options")
        print("2. Default save location")
        print("3. Prediction model settings")
        print("4. Back to main menu")
        
        choice = input("Select option (1-4): ").strip()
        if choice == '4':
            return
        else:
            print("🚧 Settings customization coming soon!")
    
    def predict_race(self, circuit: str, year: int = 2025, save: bool = False) -> pd.DataFrame:
        """Predict race with minimal setup"""
        # Normalize circuit name
        circuit_key = self.circuit_db.find_circuit(circuit)
        if not circuit_key:
            available = list(self.circuit_db.get_all_circuits().keys())
            print(f"❌ Circuit '{circuit}' not found.")
            print(f"Available circuits: {', '.join(available[:5])}...")
            print("Use --list-circuits to see all available circuits")
            return pd.DataFrame()
        
        # Get lineup
        if year >= 2025:
            lineup = self.driver_db.get_2025_lineup()
        else:
            lineup = self.driver_db.get_historical_lineup(year)
            
        if lineup is None or lineup.empty:
            print(f"❌ No lineup data available for {year}")
            return pd.DataFrame()
        
        # Add circuit info to lineup
        circuit_data = self.circuit_db.get_circuit_data(circuit_key)
        lineup['circuit'] = circuit_data['name']
        lineup['year'] = year
        
        return self._run_prediction(circuit_key, year, lineup, save=save, interactive=False)
    
    def _run_prediction(self, circuit: str, year: int, lineup: pd.DataFrame, 
                       save: bool = False, interactive: bool = False) -> pd.DataFrame:
        """Run the actual prediction"""
        try:
            circuit_data = self.circuit_db.get_circuit_data(circuit)
            
            # Generate predictions
            predictions = self.prediction_engine.predict_race(lineup, circuit_data, year)
            
            if predictions.empty:
                print("❌ No predictions generated")
                return pd.DataFrame()
            
            # Format and display results
            formatted_results = self.formatter.format_predictions(predictions, circuit_data, interactive)
            print(formatted_results)
            
            # Save if requested
            if save:
                filename = self.file_manager.save_predictions(predictions, circuit, year)
                if interactive:
                    print(f"\n💾 Predictions saved to: {filename}")
                else:
                    print(f"Saved to: {filename}")
            
            return predictions
            
        except Exception as e:
            print(f"❌ Prediction failed: {e}")
            if interactive:
                import traceback
                print("Debug info:")
                traceback.print_exc()
            return pd.DataFrame()

def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(
        description='F1 Race Predictor - Universal CLI Interface',
        epilog="""
Examples:
  python f1_predictor.py                    # Interactive mode
  python f1_predictor.py spa 2025           # Predict Spa-Francorchamps 2025
  python f1_predictor.py monaco 2025 --save # Predict Monaco 2025 and save to file
  python f1_predictor.py --list-circuits    # Show available circuits
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('circuit', nargs='?', help='Circuit name or key (e.g., spa, monaco, silverstone)')
    parser.add_argument('year', nargs='?', type=int, default=2025, help='Race year (default: 2025)')
    parser.add_argument('--interactive', '-i', action='store_true', help='Force interactive mode')
    parser.add_argument('--save', '-s', action='store_true', help='Save predictions to file')
    parser.add_argument('--list-circuits', '-l', action='store_true', help='List available circuits')
    parser.add_argument('--format', '-f', choices=['table', 'json', 'csv'], default='table', 
                       help='Output format (default: table)')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = F1Predictor()
    
    # Handle list circuits
    if args.list_circuits:
        predictor._list_circuits()
        return
    
    # Handle interactive mode
    if args.interactive or (not args.circuit):
        predictor.interactive_mode()
        return
    
    # Handle quick prediction
    if args.circuit:
        predictions = predictor.predict_race(args.circuit, args.year, save=args.save)
        if not predictions.empty and args.format != 'table':
            if args.format == 'json':
                print(predictions.to_json(indent=2))
            elif args.format == 'csv':
                print(predictions.to_csv(index=False))

if __name__ == "__main__":
    main()