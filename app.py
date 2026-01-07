from feature_engineering import FinancialFeatureEngineer

import streamlit as st
import pandas as pd
from joblib import load

# -----------------------------
# Load pipeline and data
# -----------------------------
import os
import gdown

MODEL_PATH = "bankruptcy_pipeline.joblib"
FILE_ID = "1GE5o60fZBASyZE4ofe_qljUa23V2bfTq"

if not os.path.exists(MODEL_PATH):
    print("Model file not found. Downloading from Google Drive...")
    
    gdown.download(
        f"https://drive.google.com/uc?id={FILE_ID}",
        MODEL_PATH,
        quiet=False
    )

pipeline = load(MODEL_PATH)

demo_df = pd.read_csv("./demo_companies.csv")

RAW_INPUT_COLS = ["year"] + [f"X{i}" for i in range(1, 19)]

st.set_page_config(page_title="Bankruptcy Prediction", layout="wide")

st.title("🏦 Corporate Bankruptcy Prediction")

# -----------------------------
# Show demo dataset
# -----------------------------
st.subheader("Demo Companies")
DISPLAY_COLS = ["company_name", "year"] + [f"X{i}" for i in range(1, 19)]

st.dataframe(
    demo_df[DISPLAY_COLS],
    use_container_width=True
)


# -----------------------------
# Select company
# -----------------------------
company_idx = st.selectbox(
    "Select a company (auto-fill)",
    options=demo_df.index,
    format_func=lambda i: f"{demo_df.loc[i, 'company_name']} ({demo_df.loc[i, 'year']})"
)

selected_row = demo_df.loc[company_idx]

# -----------------------------
# Input fields (auto-filled)
# -----------------------------
st.subheader("Financial Inputs")

input_values = {}
cols = st.columns(3)

for i, col in enumerate(RAW_INPUT_COLS):
    with cols[i % 3]:
        input_values[col] = st.number_input(
            label=col,
            value=float(selected_row[col])
        )

# -----------------------------
# Predict
# -----------------------------
if st.button("Predict Risk"):
    input_df = pd.DataFrame([input_values])

    pred = pipeline.predict(input_df)[0]
    prob = pipeline.predict_proba(input_df)[0][1]

   

    if pred == 'failed':
        risk_pct = prob * 100
        st.error(f"⚠️ High Risk of Bankruptcy ({risk_pct:.1f}%)")
    else:
        risk_pct = (1 - prob) * 100
        st.success(f"✅ Likely to Survive (Risk: {risk_pct:.1f}%)")
