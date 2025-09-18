# app.py (robust IDS dashboard with feature alignment)
import streamlit as st
import pandas as pd
import numpy as np
import joblib, os
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report

st.set_page_config(page_title="IoT IDS Dashboard", layout="wide")
st.title("📡 IoT Intrusion Detection System (IIDS)")

# Load models once
@st.cache_resource
def load_models():
    models = {}
    models["scaler"], models["cols"] = joblib.load("models/scaler.joblib")
    models["iso"] = joblib.load("models/iso_forest.joblib")
    models["rf"] = joblib.load("models/rf.joblib")
    models["ae"] = load_model("models/autoencoder.h5", compile=False)  # or .keras if retrained
    models["ae_thr"] = joblib.load("models/ae_threshold.joblib")
    return models

models = load_models()

# File upload
uploaded = st.sidebar.file_uploader("📂 Upload CSV for Detection", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
    st.write("📄 Uploaded Data Preview:", df.head())
else:
    st.info("No file uploaded. Using first 200 rows from processed dataset.")
    d = np.load("data/processed.npz", allow_pickle=True)
    df = pd.DataFrame(d["X"][:200], columns=list(d["cols"]))

# --- Prepare numeric data robustly ---
expected_cols = models["cols"]

# Add missing expected columns and fill with 0
missing_cols = [c for c in expected_cols if c not in df.columns]
for col in missing_cols:
    df[col] = 0

# Drop any extra columns not expected
X_df = df[expected_cols].fillna(0)

if missing_cols:
    st.warning(f"⚠️ {len(missing_cols)} features were missing in uploaded file. "
               f"Filled with 0. Missing: {missing_cols[:5]}{'...' if len(missing_cols) > 5 else ''}")

st.write(f"✅ Uploaded file had {df.shape[1]} columns, model expects {len(expected_cols)} features.")
X = models["scaler"].transform(X_df.values)

# Check if labels exist
has_labels = "Label" in df.columns
true_y = df["Label"].values if has_labels else None

# --- Run all 3 models side-by-side ---
col1, col2, col3 = st.columns(3)

# Isolation Forest
with col1:
    st.subheader("🔍 Isolation Forest")
    preds = models["iso"].predict(X)
    preds = np.where(preds == -1, 1, 0)  # anomaly = 1
    st.write("Prediction counts:", pd.Series(preds).value_counts().to_dict())
    if has_labels:
        st.text(classification_report(true_y, preds))

# Autoencoder
with col2:
    st.subheader("🤖 Autoencoder")
    recon = np.mean((X - models["ae"].predict(X))**2, axis=1)
    preds = (recon > models["ae_thr"]).astype(int)
    st.write("Prediction counts:", pd.Series(preds).value_counts().to_dict())
    if has_labels:
        st.text(classification_report(true_y, preds))

# RandomForest
with col3:
    st.subheader("🌲 RandomForest")
    preds = models["rf"].predict(X)
    st.write("Prediction counts:", pd.Series(preds).value_counts().to_dict())
    if has_labels:
        st.text(classification_report(true_y, preds))


