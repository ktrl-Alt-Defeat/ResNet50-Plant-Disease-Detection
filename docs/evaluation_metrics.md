# Evaluation & Benchmarking Suite Documentation

This document details the mathematical metrics, model evaluation engine, calibration algorithms, and latency benchmark suite implemented in [`src/evaluate.py`](file:///d:/resnet%20crop%20detection/src/evaluate.py), [`src/metrics.py`](file:///d:/resnet%20crop%20detection/src/metrics.py), and [`src/benchmark.py`](file:///d:/resnet%20crop%20detection/src/benchmark.py).

---

## 📊 Classification Evaluation Metrics (`src/metrics.py`)

### 1. Accuracy Calculations
- **Top-1 Accuracy**:
  $$\text{Top-1 Acc} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(\hat{y}_i = y_i)$$
- **Top-K Accuracy ($K=3, 5$)**:
  $$\text{Top-K Acc} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(y_i \in \text{TopK}(\mathbf{P}_i))$$

### 2. Multi-Class Precision, Recall, and F1-Score
Using `scikit-learn` integration:
- **Per-Class Precision & Recall**:
  $$P_c = \frac{TP_c}{TP_c + FP_c}, \quad R_c = \frac{TP_c}{TP_c + FN_c}$$
- **Per-Class F1-Score**:
  $$F1_c = 2 \cdot \frac{P_c \cdot R_c}{P_c + R_c}$$
- **Macro F1-Score** (Unweighted mean across all classes $C$):
  $$\text{Macro F1} = \frac{1}{C} \sum_{c=1}^C F1_c$$
- **Weighted F1-Score** (Support-weighted mean across classes):
  $$\text{Weighted F1} = \sum_{c=1}^C \frac{N_c}{N} F1_c$$

---

## 🎯 Model Calibration & ECE (`calculate_ece`)

Model calibration assesses how well prediction confidence aligns with empirical accuracy.

### Expected Calibration Error (ECE)
Predictions are grouped into $M = 10$ or $15$ equal-width confidence bins $B_m \subset (0, 1]$:
$$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$
Where:
- $\text{acc}(B_m) = \frac{1}{|B_m|} \sum_{i \in B_m} \mathbb{I}(\hat{y}_i = y_i)$
- $\text{conf}(B_m) = \frac{1}{|B_m|} \sum_{i \in B_m} \hat{p}_i$

Calibration curves are saved to `results/calibration_plot.png`.

---

## ⚡ Latency & Throughput Benchmarking Engine (`src/benchmark.py`)

The benchmark engine measures inference performance on target hardware:

1. **Warmup Phase**: Runs 50 warmup iterations to stabilize CUDA clock frequencies and memory allocations.
2. **Timing Protocol**: Uses `torch.cuda.Event(enable_timing=True)` for sub-millisecond GPU timing (or `time.perf_counter()` on CPU).
3. **Benchmarking Metrics**:
   - **Mean Latency (ms)**: Average time per batch $[B, 3, 224, 224]$.
   - **P95 & P99 Latency (ms)**: 95th and 99th percentile latency bounds.
   - **Throughput (FPS)**: Images processed per second:
     $$\text{Throughput} = \frac{\text{Batch Size} \times 1000}{\text{Mean Latency (ms)}}$$
   - **Peak VRAM Allocation**: Measured via `torch.cuda.max_memory_allocated()`.
- Benchmark results are persisted to `results/inference_benchmark.json`.
