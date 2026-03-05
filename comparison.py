# =========================================================
# ARIMA vs SLA-AWARE LSTM AUTOSCALER COMPARISON
# =========================================================

import numpy as np
import pandas as pd
import tensorflow as tf
import joblib

from sklearn.metrics import classification_report, roc_auc_score
from statsmodels.tsa.arima.model import ARIMA

# =========================================================
# LOAD MODEL + SCALER
# =========================================================

model = tf.keras.models.load_model("sla_lstm_autoscaler.keras")
scaler = joblib.load("scaler.save")

print("Models loaded.")

# =========================================================
# CONFIG
# =========================================================

UPSCALE_THRESHOLD = 0.45
DOWNSCALE_THRESHOLD = 0.35

MAX_SCALE_UP = 40
MIN_INSTANCES = 1

VM_COST = 0.05

DATASET = "sla_violation_dataset_100k_moderate_noise.csv"

TARGET = "sla_violation_future"

BASE_FEATURES = [
    "p95_latency_ms","p99_latency_ms","error_rate",
    "queue_length","queue_wait_time_ms",
    "cpu_utilization","memory_utilization",
    "delta_p95_latency","delta_queue_length",
    "latency_slope","error_rate_slope",
    "active_instances"
]

# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv(DATASET)
df = df.sort_values("timestamp").reset_index(drop=True)

ROLL = 5

df["p95_latency_roll_mean"] = df["p95_latency_ms"].rolling(ROLL).mean()
df["cpu_roll_mean"] = df["cpu_utilization"].rolling(ROLL).mean()
df["queue_roll_mean"] = df["queue_length"].rolling(ROLL).mean()

df.fillna(0,inplace=True)

FEATURE_COLUMNS = BASE_FEATURES + [
    "p95_latency_roll_mean",
    "cpu_roll_mean",
    "queue_roll_mean"
]

X_raw = df[FEATURE_COLUMNS]
y_raw = df[TARGET]

X_scaled = scaler.transform(X_raw)

# =========================================================
# BUILD SEQUENCES
# =========================================================

def build_sequences(X,y,seq_len):

    Xs=[]
    ys=[]

    for i in range(len(X)-seq_len):
        Xs.append(X[i:i+seq_len])
        ys.append(y.iloc[i+seq_len])

    return np.array(Xs),np.array(ys)


SEQ=30

X_seq,y_seq = build_sequences(X_scaled,y_raw,SEQ)

split = int(0.85*len(X_seq))

X_test = X_seq[split:]
y_test = y_seq[split:]

# =========================================================
# FUNCTION : PHYSICS SCALING
# =========================================================

def apply_scaling_physics(seq,old,new,scaler,features):

    ratio = old/new

    affected = [
        "queue_length",
        "queue_wait_time_ms",
        "cpu_utilization",
        "memory_utilization",
        "p95_latency_ms",
        "p99_latency_ms"
    ]

    for f in affected:

        idx = features.index(f)

        minv = scaler.data_min_[idx]
        maxv = scaler.data_max_[idx]

        real = seq[-1,idx]*(maxv-minv)+minv

        real *= ratio

        scaled = (real-minv)/(maxv-minv)

        seq[-1,idx] = scaled

    return seq


# =========================================================
# 1️⃣ LSTM SLA AUTOSCALER
# =========================================================

def lstm_controller(seq):

    seq = seq.copy()

    idx = FEATURE_COLUMNS.index("active_instances")

    minv = scaler.data_min_[idx]
    maxv = scaler.data_max_[idx]

    inst = seq[-1,idx]*(maxv-minv)+minv
    inst = int(round(inst))

    while True:

        prob = model.predict(seq[np.newaxis,:,:],verbose=0)[0][0]

        if prob < UPSCALE_THRESHOLD:
            break

        new_inst = inst+2

        seq = apply_scaling_physics(seq,inst,new_inst,scaler,FEATURE_COLUMNS)

        scaled = (new_inst-minv)/(maxv-minv)

        seq[-1,idx] = scaled

        inst = new_inst

        if inst>60:
            break

    return inst


# =========================================================
# 2️⃣ ARIMA AUTOSCALER
# =========================================================

def arima_controller(history,current_instances):

    try:

        model = ARIMA(history,order=(2,1,2))
        fit = model.fit()

        forecast = fit.forecast()[0]

    except:
        forecast = history[-1]

    # simple threshold rule

    if forecast > 350:
        current_instances += 2

    elif forecast < 200 and current_instances>1:
        current_instances -= 1

    return current_instances


# =========================================================
# COMPARISON LOOP
# =========================================================

print("\nRunning comparison...")

baseline_violations = 0
lstm_violations = 0
arima_violations = 0

baseline_cost = 0
lstm_cost = 0
arima_cost = 0

latency_series = df["p95_latency_ms"].values

for i,seq in enumerate(X_test[:200]):

    seq0 = seq.copy()

    # original instances

    idx = FEATURE_COLUMNS.index("active_instances")

    minv = scaler.data_min_[idx]
    maxv = scaler.data_max_[idx]

    inst = seq0[-1,idx]*(maxv-minv)+minv
    inst = int(round(inst))

    # baseline

    prob = model.predict(seq0[np.newaxis,:,:],verbose=0)[0][0]

    if prob >= UPSCALE_THRESHOLD:
        baseline_violations += 1

    baseline_cost += inst*VM_COST

    # LSTM controller

    new_inst_lstm = lstm_controller(seq0.copy())

    seq_lstm = apply_scaling_physics(seq0.copy(),inst,new_inst_lstm,scaler,FEATURE_COLUMNS)

    prob_lstm = model.predict(seq_lstm[np.newaxis,:,:],verbose=0)[0][0]

    if prob_lstm >= UPSCALE_THRESHOLD:
        lstm_violations += 1

    lstm_cost += new_inst_lstm*VM_COST

    # ARIMA controller

    history = latency_series[i:i+SEQ]

    new_inst_arima = arima_controller(history,inst)

    seq_arima = apply_scaling_physics(seq0.copy(),inst,new_inst_arima,scaler,FEATURE_COLUMNS)

    prob_arima = model.predict(seq_arima[np.newaxis,:,:],verbose=0)[0][0]

    if prob_arima >= UPSCALE_THRESHOLD:
        arima_violations += 1

    arima_cost += new_inst_arima*VM_COST


# =========================================================
# RESULTS
# =========================================================

print("\n==============================")
print("AUTOSCALER COMPARISON RESULTS")
print("==============================")

print("\nBaseline Violations:",baseline_violations)
print("LSTM SLA Violations:",lstm_violations)
print("ARIMA Violations:",arima_violations)

print("\nBaseline Cost:",round(baseline_cost,2))
print("LSTM Cost:",round(lstm_cost,2))
print("ARIMA Cost:",round(arima_cost,2))

print("\nViolation Reduction (LSTM):",
      round((baseline_violations-lstm_violations)/baseline_violations*100,2),"%")

print("Violation Reduction (ARIMA):",
      round((baseline_violations-arima_violations)/baseline_violations*100,2),"%")