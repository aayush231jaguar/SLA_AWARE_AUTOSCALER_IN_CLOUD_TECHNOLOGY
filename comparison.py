# =========================================================
# LSTM SLA-AWARE AUTOSCALER vs ARIMA AUTOSCALER
# =========================================================

import numpy as np
import pandas as pd
import tensorflow as tf
import joblib

from sklearn.metrics import classification_report, roc_auc_score
from statsmodels.tsa.arima.model import ARIMA

# ================================
# REPRODUCIBILITY
# ================================

np.random.seed(42)
tf.random.set_seed(42)

# ================================
# LOAD MODEL & SCALER
# ================================

model = tf.keras.models.load_model("sla_lstm_autoscaler.keras")
scaler = joblib.load("scaler.save")

print("Model and scaler loaded.")

# ================================
# CONFIG
# ================================

UPSCALE_THRESHOLD = 0.45
DOWNSCALE_THRESHOLD = 0.35
MAX_SCALE_UP = 40
MIN_INSTANCES = 1
VM_COST_PER_UNIT = 0.05

SEQ_LEN = 30
TEST_SAMPLES = 1000

DATASET_PATH = "sla_violation_dataset_100k_moderate_noise.csv"
TARGET_COLUMN = "sla_violation_future"

# ================================
# FEATURES
# ================================

BASE_FEATURES = [
    "p95_latency_ms","p99_latency_ms","error_rate",
    "queue_length","queue_wait_time_ms",
    "cpu_utilization","memory_utilization",
    "delta_p95_latency","delta_queue_length",
    "latency_slope","error_rate_slope",
    "active_instances"
]

# ================================
# LOAD DATA
# ================================

df = pd.read_csv(DATASET_PATH)
df = df.sort_values("timestamp").reset_index(drop=True)

ROLL_WINDOW = 5

df["p95_latency_roll_mean"] = df["p95_latency_ms"].rolling(ROLL_WINDOW).mean()
df["cpu_roll_mean"] = df["cpu_utilization"].rolling(ROLL_WINDOW).mean()
df["queue_roll_mean"] = df["queue_length"].rolling(ROLL_WINDOW).mean()

df.fillna(0,inplace=True)

FEATURE_COLUMNS = BASE_FEATURES + [
    "p95_latency_roll_mean",
    "cpu_roll_mean",
    "queue_roll_mean"
]

X_raw = df[FEATURE_COLUMNS]
y_raw = df[TARGET_COLUMN]

X_scaled = scaler.transform(X_raw)

latency_series = df["p95_latency_ms"].values

# ================================
# BUILD SEQUENCES
# ================================

def build_sequences(X,y,seq_len):

    X_seq=[]
    y_seq=[]

    for i in range(len(X)-seq_len):
        X_seq.append(X[i:i+seq_len])
        y_seq.append(y.iloc[i+seq_len])

    return np.array(X_seq),np.array(y_seq)

X_seq,y_seq = build_sequences(X_scaled,y_raw,SEQ_LEN)

split = int(0.85*len(X_seq))

X_test = X_seq[split:]
y_test = y_seq[split:]

print("Total sequences:",len(X_seq))
print("Test sequences:",len(X_test))

# ================================
# CLASSIFIER PERFORMANCE
# ================================

y_prob = model.predict(X_test)
y_pred = (y_prob>=0.5).astype(int)

print("\nClassifier Performance")
print(classification_report(y_test,y_pred))
print("ROC-AUC:",roc_auc_score(y_test,y_prob))

# ================================
# PHYSICS SCALING
# ================================

def apply_scaling_physics(seq,old_inst,new_inst,scaler,feature_cols):

    ratio = old_inst/new_inst

    affected = [
        "queue_length",
        "queue_wait_time_ms",
        "cpu_utilization",
        "memory_utilization",
        "p95_latency_ms",
        "p99_latency_ms"
    ]

    for name in affected:

        idx = feature_cols.index(name)

        minv = scaler.data_min_[idx]
        maxv = scaler.data_max_[idx]

        real_val = seq[-1,idx]*(maxv-minv)+minv
        real_val *= ratio

        scaled_val = (real_val-minv)/(maxv-minv)
        seq[-1,idx] = scaled_val

    return seq

# =========================================================
# LSTM CONTROLLER EVALUATION
# =========================================================

baseline_violations = 0
post_scaling_violations = 0

baseline_total_cost = 0
post_scaling_total_cost = 0

print("\n=== LSTM CONTROLLER EVALUATION ===")

for seq in X_test[:TEST_SAMPLES]:

    seq_original = seq.copy()

    initial_prob = model.predict(
        seq_original[np.newaxis,:,:],verbose=0
    )[0][0]

    idx = FEATURE_COLUMNS.index("active_instances")

    inst_min = scaler.data_min_[idx]
    inst_max = scaler.data_max_[idx]

    scaled_val = seq_original[-1,idx]

    original_inst = int(round(
        scaled_val*(inst_max-inst_min)+inst_min
    ))

    if initial_prob >= UPSCALE_THRESHOLD:
        baseline_violations += 1

    baseline_total_cost += original_inst*VM_COST_PER_UNIT

    seq_test = seq_original.copy()
    current_inst = original_inst

    if initial_prob >= UPSCALE_THRESHOLD:

        while True:

            prob = model.predict(
                seq_test[np.newaxis,:,:],verbose=0
            )[0][0]

            if prob < UPSCALE_THRESHOLD:
                break

            step = 2
            new_inst = current_inst + step

            seq_test = apply_scaling_physics(
                seq_test,current_inst,new_inst,
                scaler,FEATURE_COLUMNS
            )

            scaled_new = (new_inst-inst_min)/(inst_max-inst_min)
            seq_test[-1,idx] = scaled_new

            current_inst = new_inst

            if current_inst > original_inst + MAX_SCALE_UP:
                break

    final_prob = model.predict(
        seq_test[np.newaxis,:,:],verbose=0
    )[0][0]

    if final_prob >= UPSCALE_THRESHOLD:
        post_scaling_violations += 1

    post_scaling_total_cost += current_inst*VM_COST_PER_UNIT

# =========================================================
# ARIMA CONTROLLER EVALUATION
# =========================================================

arima_violations = 0
arima_total_cost = 0

print("\n=== ARIMA CONTROLLER EVALUATION ===")

for i,seq in enumerate(X_test[:TEST_SAMPLES]):

    seq_original = seq.copy()

    idx = FEATURE_COLUMNS.index("active_instances")

    inst_min = scaler.data_min_[idx]
    inst_max = scaler.data_max_[idx]

    scaled_val = seq_original[-1,idx]

    original_inst = int(round(
        scaled_val*(inst_max-inst_min)+inst_min
    ))

    seq_test = seq_original.copy()
    current_inst = original_inst

    latency_history = latency_series[i:i+SEQ_LEN]

    try:
        arima_model = ARIMA(latency_history,order=(1,1,1))
        fit = arima_model.fit()
        forecast_latency = fit.forecast()[0]
    except:
        forecast_latency = latency_history[-1]

    TARGET_LATENCY = 250

    while forecast_latency > TARGET_LATENCY:

        new_inst = current_inst + 2

        seq_test = apply_scaling_physics(
            seq_test,current_inst,new_inst,
            scaler,FEATURE_COLUMNS
        )

        scaled_new = (new_inst-inst_min)/(inst_max-inst_min)
        seq_test[-1,idx] = scaled_new

        current_inst = new_inst

        forecast_latency = forecast_latency*(original_inst/current_inst)

        if current_inst > original_inst + MAX_SCALE_UP:
            break

    final_prob = model.predict(
        seq_test[np.newaxis,:,:],verbose=0
    )[0][0]

    if final_prob >= UPSCALE_THRESHOLD:
        arima_violations += 1

    arima_total_cost += current_inst*VM_COST_PER_UNIT

# =========================================================
# FINAL RESULTS
# =========================================================

print("\n==============================")
print("AUTOSCALER COMPARISON RESULTS")
print("==============================")

print("\nBaseline Violations:",baseline_violations)
print("LSTM Violations:",post_scaling_violations)
print("ARIMA Violations:",arima_violations)

print("\nBaseline Cost:",round(baseline_total_cost,2))
print("LSTM Cost:",round(post_scaling_total_cost,2))
print("ARIMA Cost:",round(arima_total_cost,2))

print("\nViolation Reduction LSTM:",
      round((baseline_violations-post_scaling_violations)
/baseline_violations*100,2),"%")

print("Violation Reduction ARIMA:",
      round((baseline_violations-arima_violations)
/baseline_violations*100,2),"%")