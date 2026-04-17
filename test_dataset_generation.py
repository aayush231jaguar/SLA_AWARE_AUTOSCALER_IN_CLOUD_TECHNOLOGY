import numpy as np
import pandas as pd

np.random.seed(42)

N = 100000
PRED_HORIZON = 5

def create_dataset(workload_type):

    timestamps = pd.date_range(
        start="2025-01-01",
        periods=N,
        freq="min"
    )

    # =================================
    # CPU Patterns
    # =================================

    if workload_type == "stable":
        cpu = 55 + np.random.normal(0,5,N)

    elif workload_type == "burst":
        cpu = 55 + np.random.normal(0,8,N)
        spikes = np.random.choice(N,300)
        cpu[spikes] += np.random.uniform(20,35,300)

    elif workload_type == "increasing":
        trend = np.linspace(45,85,N)
        cpu = trend + np.random.normal(0,6,N)

    cpu = np.clip(cpu,10,100)

    # =================================
    # Other metrics derived from CPU
    # =================================

    memory = np.clip(cpu*0.8 + np.random.normal(0,4,N),10,100)

    queue = np.clip((cpu/8) + np.random.normal(0,2,N),0,None)

    queue_wait = queue*3 + np.random.normal(0,5,N)

    p95_latency = 120 + cpu*2 + queue*4 + np.random.normal(0,20,N)

    p99_latency = p95_latency + np.random.normal(40,15,N)

    error_rate = np.maximum(0,(cpu-80)/150) + np.random.normal(0,0.003,N)

    instances = np.clip((cpu/18).astype(int),1,None)

    # =================================
    # Temporal Features
    # =================================

    delta_latency = np.gradient(p95_latency)

    delta_queue = np.gradient(queue)

    latency_slope = np.gradient(p95_latency)

    error_slope = np.gradient(error_rate)

    # =================================
    # Risk Score (same logic as training)
    # =================================

    risk_score = (
        0.40 * (p95_latency / 500) +
        0.25 * (p99_latency / 700) +
        0.15 * (error_rate / 0.20) +
        0.10 * (queue / 40) +
        0.10 * (cpu / 100)
    )

    risk_score += np.random.normal(0,0.05,N)

    thresholds = np.random.uniform(0.42,0.55,N)

    future_risk = np.roll(risk_score,-PRED_HORIZON)

    sla_violation = (future_risk > thresholds).astype(int)

    sla_violation[-PRED_HORIZON:] = 0

    # =================================
    # Dataframe
    # =================================

    df = pd.DataFrame({

        "timestamp":timestamps,

        "p95_latency_ms":p95_latency,
        "p99_latency_ms":p99_latency,

        "error_rate":error_rate,

        "queue_length":queue,
        "queue_wait_time_ms":queue_wait,

        "cpu_utilization":cpu,
        "memory_utilization":memory,

        "delta_p95_latency":delta_latency,
        "delta_queue_length":delta_queue,

        "latency_slope":latency_slope,
        "error_rate_slope":error_slope,

        "active_instances":instances,

        "sla_violation_future":sla_violation
    })

    return df


# =================================
# Generate datasets
# =================================

stable_df = create_dataset("stable")
burst_df = create_dataset("burst")
increase_df = create_dataset("increasing")


stable_df.to_csv("sla_dataset_stable_workload.csv",index=False)
burst_df.to_csv("sla_dataset_burst_workload.csv",index=False)
increase_df.to_csv("sla_dataset_increasing_workload.csv",index=False)


print("Datasets generated")

print("\nStable SLA ratio")
print(stable_df["sla_violation_future"].value_counts(normalize=True))

print("\nBurst SLA ratio")
print(burst_df["sla_violation_future"].value_counts(normalize=True))

print("\nIncreasing SLA ratio")
print(increase_df["sla_violation_future"].value_counts(normalize=True))