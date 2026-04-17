# SLA_AWARE_AUTOSCALER_IN_CLOUD_TECHNOLOGY

## 🚀 Overview
Predictive autoscaling system using LSTM to proactively prevent SLA violations in cloud environments.

This project proposes an intelligent autoscaling framework that predicts potential SLA violations using multivariate time-series data and dynamically adjusts resources before system performance degrades.

Unlike traditional reactive autoscalers, this system anticipates future workload behavior and enables proactive scaling decisions.

---

## 🧠 Key Idea
The system models cloud performance as a **temporal sequence problem**, where future SLA violations depend on past trends in:

- Latency (p95, p99)
- CPU utilization
- Memory utilization
- Queue length and wait time

An **LSTM (Long Short-Term Memory)** network is used to capture these temporal dependencies and predict the probability of SLA violations.

---

## ⚙️ Features

- LSTM-based SLA violation prediction
- Multivariate time-series modeling
- Sliding window sequence construction
- Adaptive autoscaling (upscaling + downscaling)
- Physics-aware scaling simulation
- Comparison with ARIMA baseline
- Evaluation on multiple workload types:
  - Moderate
  - Burst
  - Periodic
  - Random

---

## 📊 Results

- **20–24% reduction in SLA violations** across moderate, periodic, and random workloads  
- **Up to 79% reduction under burst workloads**  
- Consistently outperformed ARIMA-based autoscaling

---

## 🛠️ Tech Stack

- Python  
- TensorFlow / Keras  
- NumPy, Pandas  
- Scikit-learn  
- Statsmodels (ARIMA)

---

## 🧪 How It Works

1. Generate synthetic cloud performance dataset  
2. Normalize features using Min-Max scaling  
3. Create temporal sequences (window size = 30)  
4. Train LSTM model to predict SLA violations  
5. Use prediction probability for autoscaling decisions:
   - High risk → upscale resources  
   - Low risk → safely downscale  

---

## 📌 Use Case

This system can be applied to:

- Cloud platforms (AWS, Azure, GCP)
- Microservices architectures
- Distributed systems requiring SLA guarantees

---

## 🔮 Future Work

- Advanced downscaling strategies for cost optimization  
- Reinforcement learning-based autoscaling  
- Multi-service dependency-aware scaling  
- Deployment and validation in real cloud environments  

---
