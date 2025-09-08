import streamlit as st
import pandas as pd
import subprocess
import sys
import os


st.set_page_config(page_title="F1 Future Race Predictor", layout="centered")
st.title("F1 Future Race Predictor")

st.markdown("Select a circuit and year to generate predictions using the stacked ensemble and race model.")

year = st.number_input("Year", min_value=2010, max_value=2030, value=2025, step=1)
circuit = st.text_input("Circuit", value="Spa-Francorchamps")
out_path = st.text_input("Output CSV path", value="predictions_future_race.csv")

if st.button("Run Prediction"):
    python_exec = sys.executable or 'python'
    cmd = [python_exec, os.path.join('src','predict_future_race.py'), '--year', str(year), '--circuit', circuit, '--output', out_path]
    with st.spinner('Running predictor...'):
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            st.code(result.stdout)
            if result.returncode != 0:
                st.error(result.stderr)
            else:
                st.success(f"Saved predictions to {out_path}")
                if os.path.exists(out_path):
                    df = pd.read_csv(out_path)
                    st.dataframe(df)
        except Exception as e:
            st.error(str(e))

st.markdown("Run locally: `streamlit run src/app_streamlit.py`")


