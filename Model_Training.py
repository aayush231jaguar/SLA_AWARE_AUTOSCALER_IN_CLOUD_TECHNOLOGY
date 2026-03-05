import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, roc_auc_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
# ================================
# STABILITY (DETERMINISTIC RUNS)
# ================================

import os
import random

SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
# ================================
# CONTROLLER CONFIG (HYSTERESIS)
# ================================

UPSCALE_THRESHOLD = 0.45
DOWNSCALE_THRESHOLD = 0.35
MAX_SCALE_UP = 40
MIN_INSTANCES = 1
# ================================
# 1. Load Dataset
# ================================
DATASET_PATH = "sla_violation_dataset_100k_moderate_noise.csv"
TARGET_COLUMN = "sla_violation_future"

BASE_FEATURES = [
    "p95_latency_ms", "p99_latency_ms", "error_rate",
    "queue_length", "queue_wait_time_ms",
    "cpu_utilization", "memory_utilization",
    "delta_p95_latency", "delta_queue_length",
    "latency_slope", "error_rate_slope",
    "active_instances"
]

df = pd.read_csv(DATASET_PATH)
df = df.sort_values("timestamp").reset_index(drop=True)


# ================================
# 2. Rolling Features
# ================================
ROLL_WINDOW = 5

df["p95_latency_roll_mean"] = df["p95_latency_ms"].rolling(ROLL_WINDOW).mean()
df["cpu_roll_mean"] = df["cpu_utilization"].rolling(ROLL_WINDOW).mean()
df["queue_roll_mean"] = df["queue_length"].rolling(ROLL_WINDOW).mean()

df.fillna(0, inplace=True)

FEATURE_COLUMNS = BASE_FEATURES + [
    "p95_latency_roll_mean",
    "cpu_roll_mean",
    "queue_roll_mean"
]

X_raw = df[FEATURE_COLUMNS]
y_raw = df[TARGET_COLUMN]


# ================================
# 3. Scaling
# ================================
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_raw)
joblib.dump(scaler, "scaler.save")

# ================================
# 4. Sequence Builder
# ================================
def build_sequences(X, y, seq_len):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_len):
        X_seq.append(X[i:i + seq_len])
        y_seq.append(y.iloc[i + seq_len])
    return np.array(X_seq), np.array(y_seq)

SEQ_LEN = 30
X_seq, y_seq = build_sequences(X_scaled, y_raw, SEQ_LEN)


# ================================
# 5. Split Data
# ================================
train_end = int(0.7 * len(X_seq))
val_end = int(0.85 * len(X_seq))

X_train, y_train = X_seq[:train_end], y_seq[:train_end]
X_val, y_val = X_seq[train_end:val_end], y_seq[train_end:val_end]
X_test, y_test = X_seq[val_end:], y_seq[val_end:]


# ================================
# 6. LSTM Model
# ================================
model = Sequential([
    LSTM(64, return_sequences=True,
         input_shape=(SEQ_LEN, X_train.shape[2])),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=64,
    callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
    verbose=1
)


# Save trained model
model.save("sla_lstm_autoscaler.keras")

print("Training complete. Model and scaler saved.")

#===============================
# EVALUATION
#================================
y_prob = model.predict(X_test)
y_pred = (y_prob >= 0.5).astype(int)

print("\nClassifier Performance")
print(classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))



