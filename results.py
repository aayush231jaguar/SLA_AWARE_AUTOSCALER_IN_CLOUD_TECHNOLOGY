# =========================================================
# MULTI-MODEL COMPARISON SCRIPT
# =========================================================

import numpy as np
import pandas as pd
import random
import os
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ================================
# REPRODUCIBILITY
# ================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

# ================================
# CONFIG
# ================================
DATASET_PATH = "sla_violation_dataset_100k_moderate_noise.csv"
TARGET_COLUMN = "sla_violation_future"
SEQ_LEN = 30
ROLL_WINDOW = 5

BASE_FEATURES = [
    "p95_latency_ms", "p99_latency_ms", "error_rate",
    "queue_length", "queue_wait_time_ms",
    "cpu_utilization", "memory_utilization",
    "delta_p95_latency", "delta_queue_length",
    "latency_slope", "error_rate_slope",
    "active_instances"
]

# ================================
# LOAD DATA
# ================================
df = pd.read_csv(DATASET_PATH)
df = df.sort_values("timestamp").reset_index(drop=True)

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
# SCALING
# ================================
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_raw)

# ================================
# BUILD SEQUENCES
# ================================
def build_sequences(X, y, seq_len):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_len):
        X_seq.append(X[i:i + seq_len])
        y_seq.append(y.iloc[i + seq_len])
    return np.array(X_seq), np.array(y_seq)

X_seq, y_seq = build_sequences(X_scaled, y_raw, SEQ_LEN)

# ================================
# TRAIN / TEST SPLIT
# ================================
train_end = int(0.7 * len(X_seq))
val_end = int(0.85 * len(X_seq))

X_train, y_train = X_seq[:train_end], y_seq[:train_end]
X_val, y_val = X_seq[train_end:val_end], y_seq[train_end:val_end]
X_test, y_test = X_seq[val_end:], y_seq[val_end:]

# Flatten for classical ML
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

results = []

# =========================================================
# 1️⃣ Logistic Regression
# =========================================================
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_flat, y_train)
y_prob = lr.predict_proba(X_test_flat)[:,1]
y_pred = (y_prob >= 0.5).astype(int)

results.append([
    "Logistic Regression",
    accuracy_score(y_test, y_pred),
    precision_score(y_test, y_pred),
    recall_score(y_test, y_pred),
    f1_score(y_test, y_pred),
    roc_auc_score(y_test, y_prob)
])

# =========================================================
# 2️⃣ Random Forest
# =========================================================
rf = RandomForestClassifier(n_estimators=200, random_state=SEED)
rf.fit(X_train_flat, y_train)
y_prob = rf.predict_proba(X_test_flat)[:,1]
y_pred = (y_prob >= 0.5).astype(int)

results.append([
    "Random Forest",
    accuracy_score(y_test, y_pred),
    precision_score(y_test, y_pred),
    recall_score(y_test, y_pred),
    f1_score(y_test, y_pred),
    roc_auc_score(y_test, y_prob)
])

# =========================================================
# 3️⃣ XGBoost
# =========================================================
xgb = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    random_state=SEED,
    use_label_encoder=False,
    eval_metric="logloss"
)
xgb.fit(X_train_flat, y_train)
y_prob = xgb.predict_proba(X_test_flat)[:,1]
y_pred = (y_prob >= 0.5).astype(int)

results.append([
    "XGBoost",
    accuracy_score(y_test, y_pred),
    precision_score(y_test, y_pred),
    recall_score(y_test, y_pred),
    f1_score(y_test, y_pred),
    roc_auc_score(y_test, y_prob)
])

# =========================================================
# 4️⃣ LSTM
# =========================================================
lstm_model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(SEQ_LEN, X_train.shape[2])),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(1, activation="sigmoid")
])

lstm_model.compile(optimizer="adam",
                   loss="binary_crossentropy",
                   metrics=["accuracy"])

lstm_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=8,
    batch_size=64,
    callbacks=[EarlyStopping(patience=3, restore_best_weights=True)],
    verbose=0
)

y_prob = lstm_model.predict(X_test).ravel()
y_pred = (y_prob >= 0.5).astype(int)

results.append([
    "LSTM",
    accuracy_score(y_test, y_pred),
    precision_score(y_test, y_pred),
    recall_score(y_test, y_pred),
    f1_score(y_test, y_pred),
    roc_auc_score(y_test, y_prob)
])

# =========================================================
# 5️⃣ GRU
# =========================================================
gru_model = Sequential([
    GRU(64, return_sequences=True, input_shape=(SEQ_LEN, X_train.shape[2])),
    Dropout(0.2),
    GRU(32),
    Dropout(0.2),
    Dense(1, activation="sigmoid")
])

gru_model.compile(optimizer="adam",
                  loss="binary_crossentropy",
                  metrics=["accuracy"])

gru_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=8,
    batch_size=64,
    callbacks=[EarlyStopping(patience=3, restore_best_weights=True)],
    verbose=0
)

y_prob = gru_model.predict(X_test).ravel()
y_pred = (y_prob >= 0.5).astype(int)

results.append([
    "GRU",
    accuracy_score(y_test, y_pred),
    precision_score(y_test, y_pred),
    recall_score(y_test, y_pred),
    f1_score(y_test, y_pred),
    roc_auc_score(y_test, y_prob)
])

# =========================================================
# RESULTS TABLE
# =========================================================
results_df = pd.DataFrame(results, columns=[
    "Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"
])

print("\n========== MODEL COMPARISON ==========")
print(results_df.sort_values("ROC-AUC", ascending=False))