# train_ae.py (updated)
import numpy as np, os, joblib
from tensorflow.keras import layers, models, callbacks

data = np.load('data/processed.npz', allow_pickle=True)
X = data['X']; y = data['y']
X_train = X[y == 0]   # only benign

def build_autoencoder(input_dim, latent_dim=16):
    inp = layers.Input(shape=(input_dim,))
    x = layers.Dense(128, activation='relu')(inp)
    x = layers.Dense(64, activation='relu')(x)
    z = layers.Dense(latent_dim, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(z)
    x = layers.Dense(128, activation='relu')(x)
    out = layers.Dense(input_dim, activation='linear')(x)
    ae = models.Model(inp, out)
    ae.compile(optimizer='adam', loss='mse')
    return ae

ae = build_autoencoder(X.shape[1])
cb = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
ae.fit(X_train, X_train, epochs=30, batch_size=256, validation_split=0.1, callbacks=[cb])

os.makedirs('models', exist_ok=True)

# ✅ Save in TensorFlow’s recommended format (.keras)
ae.save('models/autoencoder.keras')

# threshold
recon = np.mean((X_train - ae.predict(X_train))**2, axis=1)
thr = float(recon.mean() + 3*recon.std())
joblib.dump(thr, 'models/ae_threshold.joblib')
print("✅ Saved: autoencoder.keras + ae_threshold.joblib")
