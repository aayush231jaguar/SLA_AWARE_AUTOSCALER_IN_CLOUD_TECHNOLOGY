import numpy as np
import pandas as pd

np.random.seed(42)

N = 100_000

timestamps = pd.date_range(
    start="2025-01-01",
    periods=N,
    freq="min"
)

# ======================================
# PERIODIC LOAD (FIXED - IMPORTANT)
# ======================================
t = np.arange(N)

# Full sinusoidal range → ensures SLA crossing
load = 0.5 + 0.5 * np.sin(2 * np.pi * t / 500)

# Add moderate noise
load += np.random.normal(0, 0.05, N)

# Keep in valid bounds
load = np.clip(load, 0, 1)

# ======================================
# METRICS GENERATED FROM LOAD
# ======================================

p95_latency_ms = 200 + load * 600 + np.random.normal(0, 40, N)
p99_latency_ms = p95_latency_ms + np.random.normal(60, 25, N)

error_rate = 0.01 + load * 0.12 + np.random.normal(0, 0.01, N)
error_rate = np.clip(error_rate, 0, 0.25)

queue_length = load * 45 + np.random.normal(0, 3, N)
queue_length = np.clip(queue_length, 0, 50)

queue_wait_time_ms = load * 250 + np.random.normal(0, 30, N)
queue_wait_time_ms = np.clip(queue_wait_time_ms, 0, 400)

cpu_utilization = 40 + load * 60 + np.random.normal(0, 5, N)
cpu_utilization = np.clip(cpu_utilization, 5, 100)

memory_utilization = 50 + load * 45 + np.random.normal(0, 5, N)
memory_utilization = np.clip(memory_utilization, 10, 100)

active_instances = (2 + load * 6).astype(int)
active_instances = np.clip(active_instances, 1, 10)

# ======================================
# TEMPORAL FEATURES
# ======================================

delta_p95_latency = np.diff(p95_latency_ms, prepend=p95_latency_ms[0])
delta_queue_length = np.diff(queue_length, prepend=queue_length[0])

latency_slope = np.gradient(p95_latency_ms)
error_rate_slope = np.gradient(error_rate)

# ======================================
# HARD SLA CONDITIONS (CRITICAL)
# ======================================

sla_violation_future = (
    (p95_latency_ms > 500) |
    (p99_latency_ms > 700) |
    (error_rate > 0.10) |
    (queue_length > 35) |
    (cpu_utilization > 85)
).astype(int)

# ======================================
# DATAFRAME
# ======================================

df = pd.DataFrame({
    "timestamp": timestamps,
    "p95_latency_ms": np.clip(p95_latency_ms, 50, 1200),
    "p99_latency_ms": np.clip(p99_latency_ms, 80, 1600),
    "error_rate": error_rate,
    "queue_length": queue_length,
    "queue_wait_time_ms": queue_wait_time_ms,
    "cpu_utilization": cpu_utilization,
    "memory_utilization": memory_utilization,
    "delta_p95_latency": delta_p95_latency,
    "delta_queue_length": delta_queue_length,
    "latency_slope": latency_slope,
    "error_rate_slope": error_rate_slope,
    "active_instances": active_instances,
    "sla_violation_future": sla_violation_future
})

# ======================================
# SAVE
# ======================================

df.to_csv("sla_periodic_100k.csv", index=False)

# ======================================
# DEBUG OUTPUT
# ======================================

print("Dataset generated successfully.")
print("\nViolation Distribution:")
print(df["sla_violation_future"].value_counts())
print(df["sla_violation_future"].value_counts(normalize=True))