# train_rf.py
import numpy as np, os, joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

data = np.load('data/processed.npz', allow_pickle=True)
X, y = data['X'], data['y']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

clf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)
clf.fit(X_train, y_train)

pred = clf.predict(X_test)
print(classification_report(y_test, pred))

os.makedirs('models', exist_ok=True)
joblib.dump(clf, 'models/rf.joblib')
print("✅ Saved: models/rf.joblib")
