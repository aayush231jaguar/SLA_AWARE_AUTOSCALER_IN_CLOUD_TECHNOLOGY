import numpy as np
import pandas as pd

np.random.seed(42)

N = 100_000

timestamps = pd.date_range(
    start="2025-01-01",
    periods=N,
    freq="min"
)

# ============================
# Base Metrics (Moderate Noise)
# ============================
p95_latency_ms = np.random.normal(300, 90, N)
p99_latency_ms = p95_latency_ms + np.random.normal(70, 30, N)

error_rate = np.clip(np.random.beta(2, 25, N), 0, 0.25)
queue_length = np.clip(np.random.poisson(10, N), 0, 45)
queue_wait_time_ms = np.random.normal(80, 30, N)

cpu_utilization = np.random.normal(60, 15, N)
memory_utilization = np.random.normal(65, 12, N)
active_instances = np.random.randint(2, 7, N)

# ============================
# Derived Temporal Features
# ============================
delta_p95_latency = np.diff(p95_latency_ms, prepend=p95_latency_ms[0])
delta_queue_length = np.diff(queue_length, prepend=queue_length[0])

latency_slope = np.gradient(p95_latency_ms) + np.random.normal(0, 3, N)
error_rate_slope = np.gradient(error_rate) + np.random.normal(0, 0.005, N)

# ============================
# Risk Score (Stronger Signal)
# ============================
risk_score = (
    0.40 * (p95_latency_ms / 500) +
    0.25 * (p99_latency_ms / 700) +
    0.15 * (error_rate / 0.20) +
    0.10 * (queue_length / 40) +
    0.10 * (cpu_utilization / 100)
)

# Reduced noise (KEY CHANGE)
risk_score += np.random.normal(0, 0.07, N)

# Narrower probabilistic threshold
thresholds = np.random.uniform(0.45, 0.65, N)

sla_violation_future = (risk_score > thresholds).astype(int)

# ============================
# Assemble Dataset
# ============================
df = pd.DataFrame({
    "timestamp": timestamps,
    "p95_latency_ms": np.clip(p95_latency_ms, 50, 1200),
    "p99_latency_ms": np.clip(p99_latency_ms, 80, 1600),
    "error_rate": error_rate,
    "queue_length": queue_length,
    "queue_wait_time_ms": np.clip(queue_wait_time_ms, 0, 400),
    "cpu_utilization": np.clip(cpu_utilization, 5, 100),
    "memory_utilization": np.clip(memory_utilization, 10, 100),
    "delta_p95_latency": delta_p95_latency,
    "delta_queue_length": delta_queue_length,
    "latency_slope": latency_slope,
    "error_rate_slope": error_rate_slope,
    "active_instances": active_instances,
    "sla_violation_future": sla_violation_future
})

df.to_csv("sla_violation_dataset_100k_moderate_noise.csv", index=False)

print("Dataset generated.")
print(df["sla_violation_future"].value_counts(normalize=True))
