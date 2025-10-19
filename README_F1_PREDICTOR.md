# 🏎️ F1 Race Predictor - Universal Interface

A simplified, user-friendly F1 race prediction system that eliminates the need to create separate files for each Grand Prix.

## 🚀 Quick Start

### Method 1: Command Line (Fastest)

```bash
# Predict any race with one command
python f1_predictor.py spa 2025                 # Spa-Francorchamps 2025
python f1_predictor.py monaco 2025 --save       # Monaco 2025 (save to file)
python f1_predictor.py silverstone 2025         # British GP 2025
python f1_predictor.py cota 2025                # US GP 2025

# List all available circuits
python f1_predictor.py --list-circuits

# Get help
python f1_predictor.py --help
```

### Method 2: Interactive Mode

```bash
# Launch interactive menu
python f1_predictor.py

# Or force interactive mode
python f1_predictor.py --interactive
```

### Method 3: Web Interface (Most User-Friendly)

```bash
# Launch beautiful web interface
streamlit run src/app_streamlit.py
```

## ✨ Features

- **🏁 All 24 F1 Circuits**: Complete 2025 calendar with detailed circuit data
- **👥 2025 Driver Lineup**: Official lineup with all confirmed driver moves:
  - Lewis Hamilton → Ferrari 
  - Carlos Sainz → Williams
  - Andrea Kimi Antonelli → Mercedes
  - And more!
- **🔮 Smart Predictions**: Multi-factor prediction engine considering:
  - Driver skill and recent form
  - Team performance and strategy
  - Circuit-specific adaptations
  - Weather and safety car probabilities
- **📊 Rich Visualizations**: Interactive charts and detailed analysis
- **💾 Multiple Export Formats**: CSV, JSON, HTML reports
- **📱 Cross-Platform**: Works on macOS, Windows, Linux

## 🎯 Examples

### Quick Predictions
```bash
# Predict next few races
python f1_predictor.py las_vegas 2025    # Las Vegas GP
python f1_predictor.py qatar 2025        # Qatar GP  
python f1_predictor.py abu_dhabi 2025    # Season finale
```

### Advanced Usage
```bash
# Save predictions in different formats
python f1_predictor.py monza 2025 --save --format json
python f1_predictor.py spa 2025 --save --format html
```

### Circuit Name Flexibility
All of these work:
```bash
python f1_predictor.py spa 2025
python f1_predictor.py "spa-francorchamps" 2025
python f1_predictor.py belgium 2025
python f1_predictor.py "belgian gp" 2025
```

## 🏎️ Supported Circuits

**2025 F1 Calendar (24 races):**
- 🇧🇭 Bahrain (`bahrain`)
- 🇸🇦 Saudi Arabia (`saudi`)
- 🇦🇺 Australia (`australia`)
- 🇯🇵 Japan (`japan`)  
- 🇨🇳 China (`china`)
- 🇺🇸 Miami (`miami`)
- 🇮🇹 Imola (`imola`)
- 🇲🇨 Monaco (`monaco`)
- 🇨🇦 Canada (`canada`)
- 🇪🇸 Spain (`spain`)
- 🇦🇹 Austria (`austria`)
- 🇬🇧 Britain (`silverstone`)
- 🇭🇺 Hungary (`hungary`)
- 🇧🇪 Belgium (`spa`)
- 🇳🇱 Netherlands (`netherlands`)
- 🇮🇹 Italy (`monza`)
- 🇦🇿 Azerbaijan (`azerbaijan`)
- 🇸🇬 Singapore (`singapore`)
- 🇺🇸 USA (`cota`)
- 🇲🇽 Mexico (`mexico`)
- 🇧🇷 Brazil (`brazil`)
- 🇺🇸 Las Vegas (`las_vegas`)
- 🇶🇦 Qatar (`qatar`)
- 🇦🇪 Abu Dhabi (`abu_dhabi`)

## 📊 Output Example

```
🏁 ============================================================
🏎️  F1 RACE PREDICTION: Circuit de Spa-Francorchamps
📍 Location: Belgium
📏 Length: 7.004km
🔄 Turns: 19
🏁 ============================================================

🏆 RACE PREDICTIONS - TOP 10:
--------------------------------------------------
🥇 Max Verstappen    (Red Bull Racing ) Score: 0.876 | Win: 61.3%
🥈 Lando Norris      (McLaren       ) Score: 0.832 | Win: 58.2%
🥉 Charles Leclerc   (Ferrari       ) Score: 0.819 | Win: 57.3%
4️⃣ Lewis Hamilton    (Ferrari       ) Score: 0.801 | Win: 56.1%
5️⃣ Oscar Piastri     (McLaren       ) Score: 0.787 | Win: 55.1%
...
```

## 🔧 Installation

The system uses your existing Python environment. If you need additional packages:

```bash
pip install pandas numpy streamlit plotly
```

## 📁 File Structure

```
Neural-network/
├── f1_predictor.py          # Main CLI interface
├── f1_config.py             # Circuit and driver database  
├── f1_utils.py              # Prediction engine and utilities
├── src/
│   └── app_streamlit.py     # Web interface
└── predictions/             # Saved prediction files
```

## 🎮 Tips & Tricks

1. **Use circuit aliases**: `monaco`, `silverstone`, `spa` work just as well as full names
2. **Save important predictions**: Use `--save` flag for races you want to keep
3. **Compare predictions**: Run the same race multiple times to see prediction stability
4. **Interactive mode**: Perfect for exploring different scenarios
5. **Web interface**: Best for detailed analysis with charts and visualizations

## 🚧 What's Different

**Before:** You had to create separate files like `predict_cota_2025.py`, `predict_spa_2025.py`, etc.

**Now:** One simple command: `python f1_predictor.py cota 2025`

- ✅ No more file creation
- ✅ No more code duplication  
- ✅ Instant predictions for any circuit
- ✅ Consistent results across all GPs
- ✅ Easy to use and remember

## 🆘 Troubleshooting

**Import errors?**
```bash
# Make sure you're in the right directory
cd /Users/vishvasshiyam/Documents/Neural-network
python f1_predictor.py spa 2025
```

**Circuit not found?**
```bash
# List available circuits
python f1_predictor.py --list-circuits
```

**Need help?**
```bash
python f1_predictor.py --help
```

---

🏁 **Ready to predict some races?** Try: `python f1_predictor.py spa 2025`