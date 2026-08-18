# Evaluation & Metrics Specification — ResNet-50 Crop Disease Detection

This document provides a comprehensive mathematical and technical specification of all metrics, evaluation routines, plots, and reports implemented in [`src/metrics.py`](file:///d:/resnet%20crop%20detection/src/metrics.py), [`src/evaluate.py`](file:///d:/resnet%20crop%20detection/src/evaluate.py), and [`src/benchmark.py`](file:///d:/resnet%20crop%20detection/src/benchmark.py).

---

## 📊 Summary Table of Implemented Metrics

| Metric Name | Mathematical Definition / Calculation | Module & Function Source | Range & Ideal | Interpretation & Usage |
| :--- | :--- | :--- | :--- | :--- |
| **Top-1 Accuracy** | $\frac{1}{N} \sum_{i=1}^{N} \mathbb{I}(\hat{y}_i = y_i)$ | [`src/metrics.py:L30`](file:///d:/resnet%20crop%20detection/src/metrics.py#L30) | `[0.0, 1.0]` (Ideal: 1.0) | Standard overall classification accuracy across all test samples. |
| **Top-K Accuracy (Top-5)** | $\frac{1}{N} \sum_{i=1}^{N} \mathbb{I}(y_i \in \text{TopK}(\hat{p}_i))$ | [`src/metrics.py:L35`](file:///d:/resnet%20crop%20detection/src/metrics.py#L35) | `[0.0, 1.0]` (Ideal: 1.0) | Fraction of test samples where ground-truth index is present within top $K$ predicted probabilities. |
| **Macro Precision** | $\frac{1}{\|C\|} \sum_{c \in C} \frac{TP_c}{TP_c + FP_c}$ | [`src/metrics.py:L50`](file:///d:/resnet%20crop%20detection/src/metrics.py#L50) | `[0.0, 1.0]` (Ideal: 1.0) | Unweighted average precision across evaluated classes. Treats all classes equally regardless of frequency. |
| **Macro Recall** | $\frac{1}{\|C\|} \sum_{c \in C} \frac{TP_c}{TP_c + FN_c}$ | [`src/metrics.py:L50`](file:///d:/resnet%20crop%20detection/src/metrics.py#L50) | `[0.0, 1.0]` (Ideal: 1.0) | Unweighted average recall (sensitivity) across evaluated classes. |
| **Macro F1-Score** | $\frac{1}{\|C\|} \sum_{c \in C} \frac{2 \cdot P_c \cdot R_c}{P_c + R_c}$ | [`src/metrics.py:L50`](file:///d:/resnet%20crop%20detection/src/metrics.py#L50) | `[0.0, 1.0]` (Ideal: 1.0) | Harmonic mean of Macro Precision and Macro Recall. Key metric monitored for checkpointing. |
| **Weighted F1-Score** | $\sum_{c \in C} \frac{N_c}{N} \cdot F1_c$ | [`src/metrics.py:L62`](file:///d:/resnet%20crop%20detection/src/metrics.py#L62) | `[0.0, 1.0]` (Ideal: 1.0) | Average F1-score weighted by individual class sample support $N_c$. |
| **Per-Class Metrics** | $P_c, R_c, F1_c, N_c$ for each class $c$ | [`src/metrics.py:L74`](file:///d:/resnet%20crop%20detection/src/metrics.py#L74) | `[0.0, 1.0]` per class | Detailed breakdown for all 124 classes. Classes with $N_c=0$ receive 0.0 and are flagged `not_evaluated=True`. |
| **Macro/Weighted AUROC** | One-vs-Rest ROC Area Under Curve | [`src/metrics.py:L126`](file:///d:/resnet%20crop%20detection/src/metrics.py#L126) | `[0.5, 1.0]` (Ideal: 1.0) | Evaluates multi-class threshold-agnostic discrimination ability. |
| **Macro/Weighted AUPRC** | One-vs-Rest Precision-Recall AUC | [`src/metrics.py:L126`](file:///d:/resnet%20crop%20detection/src/metrics.py#L126) | `[0.0, 1.0]` (Ideal: 1.0) | Evaluates model ranking performance under class imbalance. |
| **ECE (15-Bin)** | $\sum_{b=1}^{B} \frac{\|B_b\|}{N} \| \text{acc}(B_b) - \text{conf}(B_b) \|$ | [`src/metrics.py:L165`](file:///d:/resnet%20crop%20detection/src/metrics.py#L165) | `[0.0, 1.0]` (Ideal: 0.0) | Expected Calibration Error measuring confidence reliability. |
| **Batch Latency (ms)** | Inference execution time per batch | [`src/benchmark.py:L9`](file:///d:/resnet%20crop%20detection/src/benchmark.py#L9) | Low (e.g. `< 50ms`) | Measured across warmup and timed iterations with `torch.cuda.synchronize()`. Reports avg, p50, p95. |
| **Throughput (img/s)** | $\frac{\text{Batch Size} \times 1000}{\text{Avg Batch Latency (ms)}}$ | [`src/benchmark.py:L77`](file:///d:/resnet%20crop%20detection/src/benchmark.py#L77) | High (e.g. `> 500 img/s`) | Total image frames processed per second during inference benchmark. |

---

## 📈 Evaluation Pipeline Architecture

Inference evaluation runs through [`src/evaluate.py`](file:///d:/resnet%20crop%20detection/src/evaluate.py:L34-L276):

1. **Model Loading & Parameter Lock**:
   - Model weights loaded from `checkpoints/best_model.pt`.
   - Initial parameter snapshot recorded to guarantee zero parameter mutation during evaluation ([`src/evaluate.py:L75`](file:///d:/resnet%20crop%20detection/src/evaluate.py#L75)).
2. **Inference Execution**:
   - Executes under `torch.no_grad()` over `test_loader`.
   - Logits passed through `torch.softmax(outputs, dim=1)` to extract class probability vectors.
3. **Non-Finite & Probability Checks**:
   - Asserts zero `NaN` or `Inf` values in logits and probabilities.
   - Asserts probability row sums equal `1.0` within `1e-3` tolerance ([`src/evaluate.py:L120-L123`](file:///d:/resnet%20crop%20detection/src/evaluate.py#L120-L123)).
4. **Latency & Throughput Benchmarking**:
   - Executes `run_standard_benchmark` with 20 warmup iterations and 100 benchmark iterations ([`src/evaluate.py:L173-L180`](file:///d:/resnet%20crop%20detection/src/evaluate.py#L173-L180)).

---

## 🖼️ Generated Visual Artifacts & CSV Reports

The evaluation pipeline automatically exports the following output files into `results/`:

| Output File Path | Format | Description & Content | Generated By Function |
| :--- | :--- | :--- | :--- |
| [`results/test_evaluation_report.json`](file:///d:/resnet%20crop%20detection/results/test_evaluation_report.json) | JSON | Master evaluation summary containing all basic metrics, macro metrics, ECE, benchmark, and sanity checks. | [`src/evaluate.py:L243`](file:///d:/resnet%20crop%20detection/src/evaluate.py#L243) |
| [`results/classification_report.csv`](file:///d:/resnet%20crop%20detection/results/classification_report.csv) | CSV | Per-class precision, recall, F1, and support plus macro and weighted summary rows. | [`src/evaluate.py:L154`](file:///d:/resnet%20crop%20detection/src/evaluate.py#L154) |
| [`results/per_class_metrics.csv`](file:///d:/resnet%20crop%20detection/results/per_class_metrics.csv) | CSV | Structured per-class breakdown with `class_index`, `class_name`, `precision`, `recall`, `f1_score`, `support`, `not_evaluated`. | [`src/evaluate.py:L145`](file:///d:/resnet%20crop%20detection/src/evaluate.py#L145) |
| `results/confusion_matrix.csv` | CSV | Raw $N \times N$ integer confusion matrix matching evaluated test classes. | [`src/metrics.py:L336`](file:///d:/resnet%20crop%20detection/src/metrics.py#L336) |
| `results/confusion_matrix.png` | PNG | High-resolution heatmap visualization of confusion matrix. | [`src/metrics.py:L349`](file:///d:/resnet%20crop%20detection/src/metrics.py#L349) |
| `results/calibration_plot.png` | PNG | 15-bin Reliability Diagram plotting predicted confidence vs. empirical accuracy. | [`src/metrics.py:L193`](file:///d:/resnet%20crop%20detection/src/metrics.py#L193) |
| `results/roc_curves.png` | PNG | One-vs-Rest Macro-Average ROC curve with AUC score. | [`src/metrics.py:L244`](file:///d:/resnet%20crop%20detection/src/metrics.py#L244) |
| `results/precision_recall_curves.png` | PNG | Multi-Class Macro-Average Precision-Recall curve with AUPRC score. | [`src/metrics.py:L284`](file:///d:/resnet%20crop%20detection/src/metrics.py#L284) |
| `results/training_curves.png` | PNG | Epoch vs. Train Loss & Validation Loss curves updated after every epoch. | [`src/train.py:L458`](file:///d:/resnet%20crop%20detection/src/train.py#L458) |
| `results/dataset_samples.png` | PNG | 4x4 visual grid of inverse-normalized dataset images with class labels. | [`src/dataset.py:L241`](file:///d:/resnet%20crop%20detection/src/dataset.py#L241) |
| `results/inference_benchmark.json` | JSON | Detailed latency breakdown (p50, p95, avg), throughput, and peak VRAM allocation. | [`src/evaluate.py:L183`](file:///d:/resnet%20crop%20detection/src/evaluate.py#L183) |

---

## 📌 Fallback Rendering Behavior

If `matplotlib` is not installed on the execution environment (`HAS_MATPLOTLIB = False`), all plot generator functions in [`src/metrics.py`](file:///d:/resnet%20crop%20detection/src/metrics.py:L23-L25) fall back seamlessly to generating plain PIL Image canvas PNG files with drawn bounding boxes and text headers, preventing runtime failures.
