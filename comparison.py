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

SEQ_LEN = 30
TEST_SAMPLES = 1000

DATASET_PATH = "random_dataset_fixed_100k.csv"
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
# METRIC STORAGE
# =========================================================

baseline_cpu=[]
baseline_ram=[]

lstm_cpu=[]
lstm_ram=[]

arima_cpu=[]
arima_ram=[]

baseline_latency=[]
lstm_latency=[]
arima_latency=[]

baseline_vms = []
lstm_vms = []
arima_vms = []


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
    baseline_vms.append(original_inst)

    cpu_idx = FEATURE_COLUMNS.index("cpu_utilization")
    ram_idx = FEATURE_COLUMNS.index("memory_utilization")
    lat_idx = FEATURE_COLUMNS.index("p95_latency_ms")

    baseline_cpu.append(seq_original[-1,cpu_idx])
    baseline_ram.append(seq_original[-1,ram_idx])
    baseline_latency.append(seq_original[-1,lat_idx])

    if initial_prob >= UPSCALE_THRESHOLD:
        baseline_violations += 1


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


    lstm_cpu.append(seq_test[-1,cpu_idx])
    lstm_ram.append(seq_test[-1,ram_idx])
    lstm_latency.append(seq_test[-1,lat_idx])
    lstm_vms.append(current_inst)


# =========================================================
# ARIMA CONTROLLER
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


    cpu_idx = FEATURE_COLUMNS.index("cpu_utilization")
    ram_idx = FEATURE_COLUMNS.index("memory_utilization")
    lat_idx = FEATURE_COLUMNS.index("p95_latency_ms")

    arima_cpu.append(seq_test[-1,cpu_idx])
    arima_ram.append(seq_test[-1,ram_idx])
    arima_latency.append(seq_test[-1,lat_idx])
    arima_vms.append(current_inst)


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

# =========================================================
# SLA COMPLIANCE RATE (NEW METRIC)
# =========================================================

total_requests = TEST_SAMPLES

baseline_compliance = (total_requests - baseline_violations) / total_requests * 100
lstm_compliance = (total_requests - post_scaling_violations) / total_requests * 100
arima_compliance = (total_requests - arima_violations) / total_requests * 100

print("\n==============================")
print("SLA COMPLIANCE RATE")
print("==============================")

print("Baseline:",round(baseline_compliance,2),"%")
print("LSTM:",round(lstm_compliance,2),"%")
print("ARIMA:",round(arima_compliance,2),"%")


# =========================================================
# FINAL COMPARISON TABLE
# =========================================================

print("\n==============================")
print("FINAL PERFORMANCE COMPARISON")
print("==============================")

baseline_vm_avg = np.mean(baseline_vms)
lstm_vm_avg = np.mean(lstm_vms)
arima_vm_avg = np.mean(arima_vms)

baseline_cpu_avg = np.mean(baseline_cpu)*100
lstm_cpu_avg = np.mean(lstm_cpu)*100
arima_cpu_avg = np.mean(arima_cpu)*100

baseline_ram_avg = np.mean(baseline_ram)*100
lstm_ram_avg = np.mean(lstm_ram)*100
arima_ram_avg = np.mean(arima_ram)*100

baseline_lat_avg = np.mean(baseline_latency)
lstm_lat_avg = np.mean(lstm_latency)
arima_lat_avg = np.mean(arima_latency)

print("\nModel Comparison:")
print("--------------------------------------------------------------")
print("Model      Violations   SLA%   CPU%   RAM%   Latency(ms)   Avg VMs")
print("--------------------------------------------------------------")

print(f"Baseline   {baseline_violations:<12} {baseline_compliance:.2f}   {baseline_cpu_avg:.2f}   {baseline_ram_avg:.2f}   {baseline_lat_avg:.2f}   {baseline_vm_avg:.2f}")
print(f"LSTM       {post_scaling_violations:<12} {lstm_compliance:.2f}   {lstm_cpu_avg:.2f}   {lstm_ram_avg:.2f}   {lstm_lat_avg:.2f}   {lstm_vm_avg:.2f}")
print(f"ARIMA      {arima_violations:<12} {arima_compliance:.2f}   {arima_cpu_avg:.2f}   {arima_ram_avg:.2f}   {arima_lat_avg:.2f}   {arima_vm_avg:.2f}")

print("--------------------------------------------------------------")


import matplotlib.pyplot as plt

models = ["Baseline","LSTM","ARIMA"]

violations = [
baseline_violations,
post_scaling_violations,
arima_violations
]

sla = [
baseline_compliance,
lstm_compliance,
arima_compliance
]

costs = [
baseline_total_cost,
post_scaling_total_cost,
arima_total_cost
]

cpu = [
baseline_cpu_avg,
lstm_cpu_avg,
arima_cpu_avg
]

latency = [
baseline_lat_avg,
lstm_lat_avg,
arima_lat_avg
]

# ================================
# SLA Compliance Chart
# ================================

plt.figure()
plt.bar(models,sla)
plt.title("SLA Compliance Comparison")
plt.ylabel("SLA Compliance (%)")
plt.show()

# ================================
# SLA Violations Chart
# ================================

plt.figure()
plt.bar(models,violations)
plt.title("SLA Violations Comparison")
plt.ylabel("Number of Violations")
plt.show()

# VM Usage Comparison
plt.figure()
plt.bar(models,[
    baseline_vm_avg,
    lstm_vm_avg,
    arima_vm_avg
])
plt.title("Average VM Usage")
plt.ylabel("Active Instances")
plt.show()

# ================================
# CPU Utilization
# ================================

plt.figure()
plt.bar(models,cpu)
plt.title("Average CPU Utilization")
plt.ylabel("CPU Utilization (%)")
plt.show()

# ================================
# Latency Comparison
# ================================

plt.figure()
plt.bar(models,latency)
plt.title("Average P95 Latency")
plt.ylabel("Latency (ms)")
plt.show()