# =========================================================
# CONTROLLER + EVALUATION SCRIPT
# =========================================================

import numpy as np
import pandas as pd
import tensorflow as tf
import joblib

from sklearn.metrics import classification_report, roc_auc_score

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

DATASET_PATH = "random_dataset_fixed_100k.csv"
TARGET_COLUMN = "sla_violation_future"

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

X_scaled = scaler.transform(X_raw)

# ================================
# SEQUENCES
# ================================
def build_sequences(X, y, seq_len):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_len):
        X_seq.append(X[i:i + seq_len])
        y_seq.append(y.iloc[i + seq_len])
    return np.array(X_seq), np.array(y_seq)

SEQ_LEN = 30
X_seq, y_seq = build_sequences(X_scaled, y_raw, SEQ_LEN)

val_end = int(0.85 * len(X_seq))
X_test, y_test = X_seq[val_end:], y_seq[val_end:]

# ================================
# CLASSIFIER EVALUATION
# ================================
y_prob = model.predict(X_test)
y_pred = (y_prob >= 0.5).astype(int)

print("\nClassifier Performance")
print(classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))

print("\nController evaluation can be added here (Section 12 logic).")


# ================================
# 8. Physics-aware Scaling
# ================================
def apply_scaling_physics(seq, old_inst, new_inst,
                          scaler, feature_cols):

    ratio = old_inst / new_inst

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

        real_val = seq[-1, idx] * (maxv - minv) + minv
        real_val *= ratio

        scaled_val = (real_val - minv) / (maxv - minv)
        seq[-1, idx] = scaled_val

    return seq


# ================================
# 9. Adaptive Autoscaling
# ================================
def adaptive_autoscaling(model, sequence, scaler, feature_cols):

    seq = sequence.copy()
    idx = feature_cols.index("active_instances")

    inst_min = scaler.data_min_[idx]
    inst_max = scaler.data_max_[idx]

    scaled_val = seq[-1, idx]
    current_inst = scaled_val * (inst_max - inst_min) + inst_min
    current_inst = int(round(current_inst))

    print("\nInitial VMs:", current_inst)

    added = 0

    while True:

        prob = model.predict(seq[np.newaxis, :, :], verbose=0)[0][0]
        print("Violation probability:", round(prob, 3))

        if prob < 0.4:
            print("✅ Safe system")
            break

        if prob < 0.6:
            step = 1
        elif prob < 0.8:
            step = 2
        else:
            step = 4

        new_inst = current_inst + step

        seq = apply_scaling_physics(
            seq, current_inst, new_inst,
            scaler, feature_cols
        )

        scaled_new = (new_inst - inst_min) / (inst_max - inst_min)
        seq[-1, idx] = scaled_new

        current_inst = new_inst
        added += step

        if added > 40:
            print("⚠ Scaling limit reached")
            break

    print("Total VMs added:", added)


# ================================
# 10. Adaptive Downscaling
# ================================
def adaptive_downscaling(model, sequence, scaler, feature_cols):

    seq = sequence.copy()
    idx = feature_cols.index("active_instances")

    inst_min = scaler.data_min_[idx]
    inst_max = scaler.data_max_[idx]

    scaled_val = seq[-1, idx]
    current_inst = scaled_val * (inst_max - inst_min) + inst_min
    current_inst = int(round(current_inst))

    print("\nStart Downscaling from:", current_inst)

    saved = 0

    while current_inst > 1:

        prob = model.predict(seq[np.newaxis, :, :], verbose=0)[0][0]
        print("Current probability:", round(prob, 3))

        if prob >= 0.4:
            print("⚠ SLA boundary reached")
            break

        if prob < 0.2:
            step = 4
        elif prob < 0.3:
            step = 2
        else:
            step = 1

        new_inst = max(1, current_inst - step)

        test_seq = apply_scaling_physics(
            seq.copy(),
            current_inst,
            new_inst,
            scaler,
            feature_cols
        )

        scaled_new = (new_inst - inst_min) / (inst_max - inst_min)
        test_seq[-1, idx] = scaled_new

        prob_test = model.predict(
            test_seq[np.newaxis, :, :],
            verbose=0
        )[0][0]

        if prob_test >= 0.4:
            print("⚠ Cannot downscale further safely")
            break

        seq = test_seq
        current_inst = new_inst
        saved += step

    print("Safe VM level:", current_inst)
    print("VMs saved:", saved)


# ================================
# 11. Smart Controller Runner
# ================================
print("\n=== Intelligent Scaling Controller ===")

for i, seq in enumerate(X_test[:20]):

    print(f"\nSequence {i}")

    # Check initial SLA probability
    prob = model.predict(seq[np.newaxis, :, :], verbose=0)[0][0]

    print("Initial SLA probability:", round(prob, 3))

    if prob >= 0.4:

        print("→ SLA risk detected → UPSCALING")
        adaptive_autoscaling(model, seq, scaler, FEATURE_COLUMNS)

    else:

        print("→ System safe → DOWNSCALING")
        adaptive_downscaling(model, seq, scaler, FEATURE_COLUMNS)

# ================================
# 12. Performance Evaluation Metrics
# ================================

VM_COST_PER_UNIT = 0.05

baseline_violations = 0
post_scaling_violations = 0

baseline_total_cost = 0
post_scaling_total_cost = 0

print("\n=== SYSTEM LEVEL EVALUATION ===")

for seq in X_test[:1000]:

    seq_original = seq.copy()

    initial_prob = model.predict(
        seq_original[np.newaxis, :, :],
        verbose=0
    )[0][0]

    idx = FEATURE_COLUMNS.index("active_instances")
    inst_min = scaler.data_min_[idx]
    inst_max = scaler.data_max_[idx]

    scaled_val = seq_original[-1, idx]
    original_inst = int(round(
        scaled_val * (inst_max - inst_min) + inst_min
    ))

    # Baseline
    if initial_prob >= UPSCALE_THRESHOLD:
        baseline_violations += 1

    baseline_total_cost += original_inst * VM_COST_PER_UNIT

    # Apply controller
    seq_test = seq_original.copy()
    current_inst = original_inst

    # ---------- UPSCALE ----------
    if initial_prob >= UPSCALE_THRESHOLD:

        while True:

            prob = model.predict(
                seq_test[np.newaxis, :, :],
                verbose=0
            )[0][0]

            if prob < UPSCALE_THRESHOLD:
                break

            step = 2
            new_inst = current_inst + step

            seq_test = apply_scaling_physics(
                seq_test,
                current_inst,
                new_inst,
                scaler,
                FEATURE_COLUMNS
            )

            scaled_new = (new_inst - inst_min) / (inst_max - inst_min)
            seq_test[-1, idx] = scaled_new

            current_inst = new_inst

            if current_inst > original_inst + MAX_SCALE_UP:
                break

    # ---------- DOWNSCALE ----------
    else:

        while current_inst > MIN_INSTANCES:

            prob = model.predict(
                seq_test[np.newaxis, :, :],
                verbose=0
            )[0][0]

            if prob >= DOWNSCALE_THRESHOLD:
                break

            new_inst = current_inst - 1

            test_seq = apply_scaling_physics(
                seq_test.copy(),
                current_inst,
                new_inst,
                scaler,
                FEATURE_COLUMNS
            )

            scaled_new = (new_inst - inst_min) / (inst_max - inst_min)
            test_seq[-1, idx] = scaled_new

            prob_test = model.predict(
                test_seq[np.newaxis, :, :],
                verbose=0
            )[0][0]

            if prob_test >= DOWNSCALE_THRESHOLD:
                break

            seq_test = test_seq
            current_inst = new_inst

    final_prob = model.predict(
        seq_test[np.newaxis, :, :],
        verbose=0
    )[0][0]

    if final_prob >= UPSCALE_THRESHOLD:
        post_scaling_violations += 1

    post_scaling_total_cost += current_inst * VM_COST_PER_UNIT
# ================================
# 13. Final Metrics
# ================================

violation_reduction = (
    (baseline_violations - post_scaling_violations)
    / baseline_violations
) * 100 if baseline_violations > 0 else 0

cost_change_percent = (
    (post_scaling_total_cost - baseline_total_cost)
    / baseline_total_cost
) * 100

print("\n========== FINAL SYSTEM METRICS ==========")
print("Baseline Violations:", baseline_violations)
print("Post-Scaling Violations:", post_scaling_violations)
print("Violation Reduction (%):", round(violation_reduction, 2))

print("\nBaseline Cost:", round(baseline_total_cost, 2))
print("Post-Scaling Cost:", round(post_scaling_total_cost, 2))
print("Cost Change (%):", round(cost_change_percent, 2))