#!/usr/bin/env python3
"""
Enhanced F1 Race Predictor - Streamlit Web Interface
User-friendly web app for predicting F1 race outcomes
"""

import streamlit as st
import pandas as pd
import sys
import os
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# Add parent directory to path to import our modules
sys.path.append('..')
sys.path.append('.')

try:
    from f1_config import CircuitDatabase, DriverDatabase
    from f1_utils import PredictionEngine, ResultFormatter, FileManager
except ImportError:
    st.error("❌ Could not import F1 prediction modules. Please ensure f1_config.py and f1_utils.py are available.")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="🏎️ F1 Race Predictor",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize components
@st.cache_resource
def get_databases():
    """Initialize databases with caching"""
    return CircuitDatabase(), DriverDatabase()

circuit_db, driver_db = get_databases()
engine = PredictionEngine()
formatter = ResultFormatter()
file_manager = FileManager()

# Main title and description
st.title("🏎️ F1 Race Predictor")
st.markdown("""
**Universal F1 Race Prediction System**  
Predict any Grand Prix with the latest 2025 driver lineup and comprehensive circuit data.
""")

# Sidebar for inputs
st.sidebar.header("🏁 Race Configuration")

# Year selection
year = st.sidebar.selectbox(
    "📅 Select Year",
    options=[2024, 2025, 2026],
    index=1,  # Default to 2025
    help="Select the racing year for predictions"
)

# Circuit selection
circuits = circuit_db.get_all_circuits()
circuit_options = [(key, data['name']) for key, data in circuits.items()]
circuit_names = [f"{data['name']} ({data['country']})" for key, data in circuits.items()]
circuit_keys = [key for key, data in circuits.items()]

selected_circuit_display = st.sidebar.selectbox(
    "🏁 Select Circuit",
    options=circuit_names,
    index=circuit_keys.index('spa') if 'spa' in circuit_keys else 0,
    help="Choose the circuit for prediction"
)

# Get the circuit key from the selection
selected_circuit_idx = circuit_names.index(selected_circuit_display)
selected_circuit = circuit_keys[selected_circuit_idx]
selected_circuit_data = circuits[selected_circuit]

# Display circuit information
st.sidebar.markdown("---")
st.sidebar.markdown("**🛣️ Circuit Information:**")
st.sidebar.write(f"**Location:** {selected_circuit_data['city']}, {selected_circuit_data['country']}")
st.sidebar.write(f"**Length:** {selected_circuit_data['length_km']:.3f} km")
st.sidebar.write(f"**Turns:** {selected_circuit_data['turns']}")
st.sidebar.write(f"**Overtaking Difficulty:** {selected_circuit_data['overtaking_difficulty']}/10")

# Advanced options
with st.sidebar.expander("⚙️ Advanced Options"):
    save_predictions = st.checkbox("💾 Save predictions to file", value=True)
    show_detailed_analysis = st.checkbox("📊 Show detailed analysis", value=True)
    export_format = st.selectbox(
        "📄 Export Format",
        options=["CSV", "JSON", "HTML"],
        help="Choose the export format for saved predictions"
    )

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    # Prediction button
    if st.button("🚀 Generate Prediction", type="primary", use_container_width=True):
        with st.spinner(f'🔮 Predicting {selected_circuit_data["name"]} {year}...'):
            try:
                # Get lineup
                if year >= 2025:
                    lineup = driver_db.get_2025_lineup()
                else:
                    lineup = driver_db.get_historical_lineup(year)
                
                if lineup is None or lineup.empty:
                    st.error(f"❌ No lineup data available for {year}")
                else:
                    # Add circuit and year info
                    lineup['circuit'] = selected_circuit_data['name']
                    lineup['year'] = year
                    
                    # Generate predictions
                    predictions = engine.predict_race(lineup, selected_circuit_data, year)
                    
                    if predictions.empty:
                        st.error("❌ No predictions generated")
                    else:
                        # Display success message
                        st.success(f"✅ Predictions generated for {selected_circuit_data['name']} {year}!")
                        
                        # Store predictions in session state
                        st.session_state.predictions = predictions
                        st.session_state.circuit_data = selected_circuit_data
                        st.session_state.prediction_year = year
                        
                        # Save to file if requested
                        if save_predictions:
                            if export_format == "JSON":
                                filename = file_manager.export_predictions(predictions, 'json', selected_circuit, year)
                            elif export_format == "HTML":
                                filename = file_manager.export_predictions(predictions, 'html', selected_circuit, year)
                            else:
                                filename = file_manager.save_predictions(predictions, selected_circuit, year)
                            
                            st.info(f"💾 Saved as: {filename}")
                        
            except Exception as e:
                st.error(f"❌ Prediction failed: {str(e)}")
                st.exception(e)

with col2:
    # Driver lineup display
    st.subheader("👥 Driver Lineup")
    
    if year >= 2025:
        lineup_display = driver_db.get_2025_lineup()
        st.write("**2025 F1 Season**")
    else:
        lineup_display = driver_db.get_historical_lineup(year)
        st.write(f"**{year} F1 Season**")
    
    if lineup_display is not None and not lineup_display.empty:
        # Group by team and display
        teams = lineup_display.groupby('team_name')
        for team_name, drivers in teams:
            with st.expander(f"🏎️ {team_name}", expanded=False):
                for _, driver in drivers.iterrows():
                    st.write(f"• {driver['driver_name']}")
    else:
        st.warning(f"⚠️ No lineup data for {year}")

# Display predictions if available
if 'predictions' in st.session_state:
    st.markdown("---")
    st.header("🏆 Race Predictions")
    
    predictions = st.session_state.predictions
    circuit_data = st.session_state.circuit_data
    prediction_year = st.session_state.prediction_year
    
    # Top 10 predictions table
    st.subheader("📊 Top 10 Predictions")
    
    display_df = predictions.head(10)[[
        'position_prediction', 'driver_name', 'team_name', 
        'performance_score', 'win_probability', 'podium_probability', 'top5_probability'
    ]].copy()
    
    # Format for display
    display_df['performance_score'] = display_df['performance_score'].round(3)
    display_df['win_probability'] = (display_df['win_probability'] * 100).round(1).astype(str) + '%'
    display_df['podium_probability'] = (display_df['podium_probability'] * 100).round(1).astype(str) + '%'
    display_df['top5_probability'] = (display_df['top5_probability'] * 100).round(1).astype(str) + '%'
    
    # Rename columns for display
    display_df.columns = ['Pos', 'Driver', 'Team', 'Score', 'Win %', 'Podium %', 'Top 5 %']
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    # Visualization section
    if show_detailed_analysis:
        st.subheader("📈 Analysis & Visualizations")
        
        # Create columns for charts
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            # Win probability chart
            st.write("**🎯 Win Probabilities**")
            top_5_win = predictions.head(8)
            fig1 = px.bar(
                top_5_win, 
                x='driver_name', 
                y='win_probability',
                title="Top 8 Win Probabilities",
                color='win_probability',
                color_continuous_scale='Viridis'
            )
            fig1.update_xaxes(tickangle=45)
            fig1.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig1, use_container_width=True)
        
        with chart_col2:
            # Team performance chart
            st.write("**🏎️ Team Performance**")
            team_avg = predictions.groupby('team_name')['performance_score'].mean().reset_index()
            team_avg = team_avg.sort_values('performance_score', ascending=True)
            
            fig2 = px.bar(
                team_avg,
                x='performance_score',
                y='team_name',
                orientation='h',
                title="Average Team Performance Score",
                color='performance_score',
                color_continuous_scale='RdYlGn'
            )
            fig2.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)
        
        # Circuit insights
        st.subheader("🛣️ Circuit Insights")
        
        insight_col1, insight_col2, insight_col3 = st.columns(3)
        
        with insight_col1:
            difficulty = circuit_data.get('overtaking_difficulty', 5)
            st.metric(
                "Overtaking Difficulty",
                f"{difficulty}/10",
                help="Higher values mean overtaking is more difficult"
            )
        
        with insight_col2:
            safety_car = circuit_data.get('safety_car_prob', 0.3) * 100
            st.metric(
                "Safety Car Probability", 
                f"{safety_car:.0f}%",
                help="Likelihood of safety car deployment during the race"
            )
        
        with insight_col3:
            weather = circuit_data.get('weather_variability', 0.5) * 100
            st.metric(
                "Weather Variability", 
                f"{weather:.0f}%",
                help="How variable weather conditions are at this circuit"
            )
        
        # Additional insights
        st.write("**📋 Key Insights:**")
        
        # Find top performers
        top_driver = predictions.iloc[0]
        top_team = predictions.groupby('team_name')['performance_score'].mean().idxmax()
        best_form = predictions.loc[predictions['recent_form'].idxmax()]
        
        st.write(f"• **Race Favorite:** {top_driver['driver_name']} ({top_driver['team_name']}) with {top_driver['win_probability']:.1%} win chance")
        st.write(f"• **Strongest Team:** {top_team} with highest average performance score")
        st.write(f"• **Best Recent Form:** {best_form['driver_name']} showing excellent recent performance")
        
        # Circuit-specific insights
        if difficulty >= 8:
            st.write(f"• **Strategy Note:** High overtaking difficulty - qualifying position will be crucial")
        elif difficulty <= 3:
            st.write(f"• **Strategy Note:** Many overtaking opportunities - exciting race expected")
        
        if safety_car >= 60:
            st.write(f"• **Strategy Note:** High safety car probability - strategy will be key")

# Recent predictions section
st.markdown("---")
st.subheader("📂 Recent Predictions")

recent_preds = file_manager.get_recent_predictions(5)

if recent_preds:
    for i, pred in enumerate(recent_preds):
        col1, col2, col3, col4 = st.columns([3, 2, 1, 1])
        
        with col1:
            st.write(f"**{pred['filename']}**")
        with col2:
            st.write(pred['date'].strftime('%Y-%m-%d %H:%M'))
        with col3:
            st.write(f"{pred['size_kb']} KB")
        with col4:
            if st.button("View", key=f"view_{i}"):
                # Load and display prediction
                loaded_pred = file_manager.load_prediction(pred['filename'])
                if loaded_pred is not None:
                    st.dataframe(loaded_pred.head(10), use_container_width=True)
else:
    st.info("No recent predictions found. Generate some predictions to see them here!")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🏁 <strong>F1 Race Predictor</strong> • Built with Streamlit • 
    <a href='https://github.com/yourusername/f1-predictor' target='_blank'>GitHub</a></p>
</div>
""", unsafe_allow_html=True)


