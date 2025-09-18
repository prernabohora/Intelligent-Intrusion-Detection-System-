# src/train_isolation.py
import numpy as np, joblib, os
from sklearn.ensemble import IsolationForest

data = np.load('data/processed.npz', allow_pickle=True)
X = data['X']
print("Training IsolationForest on X shape", X.shape)

clf = IsolationForest(
    n_estimators=200,
    contamination=0.02,
    random_state=42,
    n_jobs=-1
)
clf.fit(X)

os.makedirs('models', exist_ok=True)
joblib.dump(clf, 'models/iso_forest.joblib')
print("Saved model -> models/iso_forest.joblib")
