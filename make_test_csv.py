# make_test_csv.py
# Usage: python make_test_csv.py
import os, joblib, numpy as np, pandas as pd

# Path to your saved scaler (you already have this)
SCALER_PATH = os.path.join("models", "scaler.joblib")
OUT_DIR = os.path.join("Dataset")   # will save into D:\PRERNA\POOLLOO\Dataset
OUT_FN = os.path.join(OUT_DIR, "test_iids_sample.csv")

if not os.path.exists(SCALER_PATH):
    raise SystemExit("Scaler not found at models/scaler.joblib - run preprocessing first.")

scaler, cols = joblib.load(SCALER_PATH)   # scaler is StandardScaler, cols is list of feature names

n_benign = 10
n_attack = 6

# Create benign samples: sample from standard normal in scaled space, then inverse-transform to original feature space
rng = np.random.default_rng(42)
Xb_scaled = rng.normal(loc=0.0, scale=1.0, size=(n_benign, len(cols)))
Xb = scaler.inverse_transform(Xb_scaled)

# Create attack samples: start from benign but add shifts to some features to simulate anomalies
Xa_scaled = rng.normal(loc=0.0, scale=1.0, size=(n_attack, len(cols)))
# add stronger signals on a few columns (first 3 features) to simulate attack behavior
if len(cols) >= 3:
    Xa_scaled[:, 0] += 6.0  # large positive on feature 0
    Xa_scaled[:, 1] -= 5.0  # large negative on feature 1
    Xa_scaled[:, 2] += 4.0
Xa = scaler.inverse_transform(Xa_scaled)

# Build dataframe and add label column (0 = benign, 1 = attack) - optional for supervised testing
df_b = pd.DataFrame(Xb, columns=cols)
df_b['Label'] = 0
df_a = pd.DataFrame(Xa, columns=cols)
df_a['Label'] = 1

df = pd.concat([df_b, df_a], ignore_index=True)
os.makedirs(OUT_DIR, exist_ok=True)
df.to_csv(OUT_FN, index=False)
print(f"Saved test CSV to: {OUT_FN}")
print("Columns written:", cols + ['Label'])
print("Rows: benign", n_benign, "attack", n_attack)
