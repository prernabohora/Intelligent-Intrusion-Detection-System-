# data.py
import os, glob
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

# ✅ Ensure dataset is downloaded via kagglehub
try:
    import kagglehub
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "kagglehub"])
    import kagglehub

DATASET_DIR = r"D:\PRERNA\POOLLOO\Dataset"

if not os.path.exists(DATASET_DIR) or not os.listdir(DATASET_DIR):
    print("Downloading N-BaIoT dataset via kagglehub...")
    path = kagglehub.dataset_download("mkashifn/nbaiot-dataset")
    print("KaggleHub downloaded to:", path)

    import shutil
    if os.path.exists(DATASET_DIR):
        shutil.rmtree(DATASET_DIR)
    shutil.copytree(path, DATASET_DIR)

print("Dataset ready at:", DATASET_DIR)

# 🔹 Preprocessing functions
def infer_label_from_filename(fname):
    if "benign" in fname.lower():
        return 0
    return 1

def load_all_csvs(data_dir, sample_frac=None, max_rows=None):
    files = glob.glob(os.path.join(data_dir, "*.csv"))
    if not files:
        raise FileNotFoundError("No CSV files in " + data_dir)
    dfs, ys = [], []
    for f in files:
        print("Reading:", f)
        df = pd.read_csv(f, low_memory=False)
        y = np.full(len(df), infer_label_from_filename(f), dtype=int)
        if sample_frac is not None:
            df = df.sample(frac=sample_frac, random_state=42)
            y = np.full(len(df), infer_label_from_filename(f), dtype=int)
        if max_rows and len(df) > max_rows:
            df = df.head(max_rows)
            y = np.full(len(df), infer_label_from_filename(f), dtype=int)
        dfs.append(df)
        ys.append(y)
    return pd.concat(dfs, ignore_index=True), np.concatenate(ys)

def clean_and_scale(df, out_scaler="models/scaler.joblib"):
    df = df.select_dtypes(include=[np.number])
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    df = df.loc[:, df.nunique() > 1]
    scaler = StandardScaler()
    X = scaler.fit_transform(df.values)
    os.makedirs(os.path.dirname(out_scaler), exist_ok=True)
    joblib.dump((scaler, list(df.columns)), out_scaler)
    return X, list(df.columns)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--sample_frac", type=float, default=0.2)  # default: 20% data
    p.add_argument("--max_rows", type=int, default=None)
    p.add_argument("--out", default="data/processed.npz")
    args = p.parse_args()

    df, y = load_all_csvs(DATASET_DIR, sample_frac=args.sample_frac, max_rows=args.max_rows)
    print("Shape:", df.shape, "Labels:", np.unique(y, return_counts=True))
    X, cols = clean_and_scale(df)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(args.out, X=X, y=y, cols=np.array(cols, dtype=object))
    print("Saved:", args.out)
