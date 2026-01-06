import gradio as gr
import pandas as pd
import numpy as np
from joblib import load

# Load model bundle
obj = load("rf_bundle.joblib")
model = obj["model"]
FEATURES = obj["feature_names"]  # final feature order expected by model

# Load dataset for sample rows
df = pd.read_csv("/kaggle/input/your_dataset.csv")  # update path
RAW_FEATURES = [
    "Net Income",
    "Total Revenue",
    "Net sales",
    "Gross Profit",
    "Total assets",
    "Total Liabilities",
    "Current assets",
    "Total Current Liabilities",
    "Inventory",
    "Cost of goods sold",
    "EBIT",
    "EBITDA",
    "Market value",
    "Total Receivables",
    "Retained Earnings",
    "Total Operating Expenses",
    "Total Long-term debt",
]

sample_df = df[RAW_FEATURES].head(10)

def compute_ratios(d):
    d["Net Profit Margin"] = d["Net Income"] / d["Total Revenue"]
    d["Gross Profit Margin"] = d["Gross Profit"] / d["Net sales"]
    d["ROA"] = d["Net Income"] / d["Total assets"]
    d["ROS"] = d["Net Income"] / d["Net sales"]
    d["Current Ratio"] = d["Current assets"] / d["Total Current Liabilities"]
    d["Quick Ratio"] = (d["Current assets"] - d["Inventory"]) / d["Total Current Liabilities"]
    d["Debt to asset ratio"] = d["Total Liabilities"] / d["Total assets"]
    return d

def predict_from_raw(*inputs):
    raw_df = pd.DataFrame([inputs], columns=RAW_FEATURES)
    full_df = compute_ratios(raw_df.copy())

    # ensure all features exist
    for f in FEATURES:
        if f not in full_df.columns:
            full_df[f] = 0.0

    full_df = full_df[FEATURES].replace([np.inf, -np.inf], 0).fillna(0)

    pred = model.predict(full_df)[0]
    prob = model.predict_proba(full_df)[0][1]

    return ("⚠️ High Risk of Failure" if pred == 0 else "✅ Likely to Survive") + f" (Prob: {prob:.2f})"

def load_row(idx):
    row = sample_df.iloc[int(idx)]
    return list(row.values)

with gr.Blocks() as app:
    gr.Markdown("## Corporate Bankruptcy Risk Prediction")

    gr.Markdown("### Sample Data (Select a Row)")
    gr.Dataframe(sample_df, interactive=False)

    row_select = gr.Dropdown(choices=[str(i) for i in range(len(sample_df))], label="Row Index")
    inputs = [gr.Number(label=f) for f in RAW_FEATURES]

    row_select.change(fn=load_row, inputs=row_select, outputs=inputs)

    gr.Markdown("### Enter / Edit Raw Financial Inputs")
    out = gr.Textbox(label="Prediction")
    gr.Button("Predict").click(fn=predict_from_raw, inputs=inputs, outputs=out)

app.launch()
