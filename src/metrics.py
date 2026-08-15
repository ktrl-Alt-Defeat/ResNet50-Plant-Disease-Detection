import json
import csv
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional, Union

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve
)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from PIL import Image, ImageDraw, ImageFont


def calculate_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate standard top-1 classification accuracy."""
    return float(accuracy_score(y_true, y_pred))


def calculate_top_k_accuracy(y_true: np.ndarray, y_prob: np.ndarray, k: int = 5) -> float:
    """
    Calculate Top-K accuracy score over predictions.
    Checks if ground-truth class index is within the top-K probability rank.
    """
    if y_prob.ndim != 2:
        raise ValueError("y_prob must be a 2D array of class probabilities.")
    num_classes = y_prob.shape[1]
    effective_k = min(k, num_classes)
    
    top_k_preds = np.argsort(y_prob, axis=1)[:, -effective_k:]
    correct = np.any(top_k_preds == y_true[:, None], axis=1)
    return float(np.mean(correct))


def calculate_macro_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: List[int]) -> Dict[str, float]:
    """Calculate unweighted macro precision, recall, and F1 across present labels."""
    prec = precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    return {
        "macro_precision": float(prec),
        "macro_recall": float(rec),
        "macro_f1": float(f1)
    }


def calculate_weighted_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: List[int]) -> Dict[str, float]:
    """Calculate weighted precision, recall, and F1 across present labels based on support."""
    prec = precision_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    rec = recall_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    return {
        "weighted_precision": float(prec),
        "weighted_recall": float(rec),
        "weighted_f1": float(f1)
    }


def calculate_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_to_idx: Dict[str, int]
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, float]]]:
    """
    Compute per-class precision, recall, F1, and support for all 124 classes in class_to_idx.
    Classes absent in test set have support=0, precision=0.0, recall=0.0, f1=0.0, and not_evaluated=True.
    """
    sorted_classes = sorted(class_to_idx.items(), key=lambda x: x[1])
    per_class_list = []
    summary_dict = {}

    for class_name, class_idx in sorted_classes:
        mask_true = (y_true == class_idx)
        support = int(np.sum(mask_true))

        if support == 0:
            prec = 0.0
            rec = 0.0
            f1 = 0.0
            not_evaluated = True
        else:
            tp = int(np.sum((y_true == class_idx) & (y_pred == class_idx)))
            fp = int(np.sum((y_true != class_idx) & (y_pred == class_idx)))
            fn = int(np.sum((y_true == class_idx) & (y_pred != class_idx)))

            prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
            not_evaluated = False

        row = {
            "class_index": class_idx,
            "class_name": class_name,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "support": support,
            "not_evaluated": not_evaluated
        }
        per_class_list.append(row)
        summary_dict[class_name] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "support": support
        }

    return per_class_list, summary_dict


def calculate_auroc_auprc(y_true: np.ndarray, y_prob: np.ndarray, present_indices: List[int]) -> Dict[str, Any]:
    """
    Calculate One-vs-Rest Macro & Weighted AUROC and Average Precision (AUPRC)
    over classes present in the test set.
    """
    try:
        # One-hot encode ground truth for present classes
        num_present = len(present_indices)
        y_true_onehot = np.zeros((len(y_true), y_prob.shape[1]), dtype=np.float32)
        for i, idx in enumerate(y_true):
            y_true_onehot[i, idx] = 1.0

        # Subset to present indices
        y_true_sub = y_true_onehot[:, present_indices]
        y_prob_sub = y_prob[:, present_indices]

        macro_auroc = float(roc_auc_score(y_true_sub, y_prob_sub, average="macro", multi_class="ovr"))
        weighted_auroc = float(roc_auc_score(y_true_sub, y_prob_sub, average="weighted", multi_class="ovr"))
        
        macro_auprc = float(average_precision_score(y_true_sub, y_prob_sub, average="macro"))
        weighted_auprc = float(average_precision_score(y_true_sub, y_prob_sub, average="weighted"))

        return {
            "macro_auroc": round(macro_auroc, 4),
            "weighted_auroc": round(weighted_auroc, 4),
            "macro_auprc": round(macro_auprc, 4),
            "weighted_auprc": round(weighted_auprc, 4),
            "status": "COMPUTED"
        }
    except Exception as e:
        return {
            "macro_auroc": "not_available",
            "weighted_auroc": "not_available",
            "macro_auprc": "not_available",
            "weighted_auprc": "not_available",
            "status": f"UNCOMPUTABLE ({e})"
        }


def calculate_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
    """
    Calculate Expected Calibration Error (ECE) with equal-width confidence binning.
    
    ECE = sum_{b=1}^B (|B_b| / N) * |acc(B_b) - conf(B_b)|
    """
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    accuracies = (predictions == y_true).astype(float)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = (confidences > bin_lower) & (confidences <= bin_upper) if i > 0 else (confidences >= bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin

    return float(ece)


def plot_calibration_curve(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15, save_png: str = "results/calibration_plot.png") -> None:
    """Generate 15-bin Reliability Diagram (Confidence vs. Accuracy)."""
    out_p = Path(save_png)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    if HAS_MATPLOTLIB:
        confidences = np.max(y_prob, axis=1)
        predictions = np.argmax(y_prob, axis=1)
        accuracies = (predictions == y_true).astype(float)

        bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
        bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2.0

        bin_accs = []
        bin_confs = []

        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper) if i > 0 else (confidences >= bin_lower) & (confidences <= bin_upper)
            if np.sum(in_bin) > 0:
                bin_accs.append(np.mean(accuracies[in_bin]))
                bin_confs.append(np.mean(confidences[in_bin]))
            else:
                bin_accs.append(0.0)
                bin_confs.append(bin_centers[i])

        plt.figure(figsize=(7, 6))
        plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration (Ideal)", linewidth=1.5)
        plt.plot(bin_confs, bin_accs, "s-", color="#1f77b4", label="ResNet-50 Model", linewidth=2, markersize=6)
        plt.xlabel("Confidence (Predicted Probability)", fontsize=11)
        plt.ylabel("Accuracy (Actual Correctness)", fontsize=11)
        plt.title("Model Calibration Reliability Diagram (15 Bins)", fontsize=12, pad=12)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(fontsize=10, loc="upper left")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.savefig(out_p, dpi=200, bbox_inches="tight")
        plt.close()
    else:
        w, h = 700, 600
        canvas = Image.new("RGB", (w, h), color=(255, 255, 255))
        draw = ImageDraw.Draw(canvas)
        draw.rectangle([60, 60, w - 60, h - 60], outline=(100, 100, 100), width=1)
        draw.line([(60, h - 60), (w - 60, 60)], fill=(0, 0, 0), width=2)
        draw.text((70, 20), "Model Calibration Reliability Diagram (15 Bins)", fill=(0, 0, 0))
        canvas.save(out_p, "PNG")

    print(f"[Metrics Plot] Saved calibration plot to: {out_p}")


def plot_roc_curves(y_true: np.ndarray, y_prob: np.ndarray, present_indices: List[int], save_png: str = "results/roc_curves.png") -> None:
    """Generate One-vs-Rest Macro-Average ROC curve plot."""
    out_p = Path(save_png)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    if HAS_MATPLOTLIB:
        y_true_onehot = np.zeros((len(y_true), y_prob.shape[1]), dtype=np.float32)
        for i, idx in enumerate(y_true):
            y_true_onehot[i, idx] = 1.0

        y_true_sub = y_true_onehot[:, present_indices]
        y_prob_sub = y_prob[:, present_indices]

        fpr, tpr, _ = roc_curve(y_true_sub.ravel(), y_prob_sub.ravel())
        macro_auc = roc_auc_score(y_true_sub, y_prob_sub, average="macro", multi_class="ovr")

        plt.figure(figsize=(7, 6))
        plt.plot(fpr, tpr, color="#2ca02c", linewidth=2, label=f"Macro-Average ROC (AUC = {macro_auc:.4f})")
        plt.plot([0, 1], [0, 1], "k--", linewidth=1.2, label="Random Classifier (AUC = 0.5000)")
        plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
        plt.ylabel("True Positive Rate (Sensitivity)", fontsize=11)
        plt.title("Multi-Class One-vs-Rest ROC Curve", fontsize=12, pad=12)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(fontsize=10, loc="lower right")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.savefig(out_p, dpi=200, bbox_inches="tight")
        plt.close()
    else:
        w, h = 700, 600
        canvas = Image.new("RGB", (w, h), color=(255, 255, 255))
        draw = ImageDraw.Draw(canvas)
        draw.rectangle([60, 60, w - 60, h - 60], outline=(100, 100, 100), width=1)
        draw.line([(60, h - 60), (w - 60, 60)], fill=(0, 0, 0), width=1)
        draw.text((70, 20), "Multi-Class One-vs-Rest ROC Curve", fill=(0, 0, 0))
        canvas.save(out_p, "PNG")

    print(f"[Metrics Plot] Saved ROC curve plot to: {out_p}")


def plot_pr_curves(y_true: np.ndarray, y_prob: np.ndarray, present_indices: List[int], save_png: str = "results/precision_recall_curves.png") -> None:
    """Generate One-vs-Rest Macro-Average Precision-Recall curve plot."""
    out_p = Path(save_png)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    if HAS_MATPLOTLIB:
        y_true_onehot = np.zeros((len(y_true), y_prob.shape[1]), dtype=np.float32)
        for i, idx in enumerate(y_true):
            y_true_onehot[i, idx] = 1.0

        y_true_sub = y_true_onehot[:, present_indices]
        y_prob_sub = y_prob[:, present_indices]

        precision, recall, _ = precision_recall_curve(y_true_sub.ravel(), y_prob_sub.ravel())
        macro_auprc = average_precision_score(y_true_sub, y_prob_sub, average="macro")

        plt.figure(figsize=(7, 6))
        plt.plot(recall, precision, color="#d62728", linewidth=2, label=f"Macro-Average PR (AUPRC = {macro_auprc:.4f})")
        plt.xlabel("Recall (Sensitivity)", fontsize=11)
        plt.ylabel("Precision (Positive Predictive Value)", fontsize=11)
        plt.title("Multi-Class Precision-Recall Curve", fontsize=12, pad=12)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(fontsize=10, loc="lower left")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.savefig(out_p, dpi=200, bbox_inches="tight")
        plt.close()
    else:
        w, h = 700, 600
        canvas = Image.new("RGB", (w, h), color=(255, 255, 255))
        draw = ImageDraw.Draw(canvas)
        draw.rectangle([60, 60, w - 60, h - 60], outline=(100, 100, 100), width=1)
        draw.text((70, 20), "Multi-Class Precision-Recall Curve", fill=(0, 0, 0))
        canvas.save(out_p, "PNG")

    print(f"[Metrics Plot] Saved Precision-Recall curve plot to: {out_p}")


def generate_confusion_matrix_artifacts(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names_present: List[str],
    save_csv: str = "results/confusion_matrix.csv",
    save_png: str = "results/confusion_matrix.png"
) -> np.ndarray:
    """
    Generate 122 x 122 confusion matrix CSV and high-resolution PNG heatmap for evaluated test classes.
    """
    present_indices = sorted(list(set(y_true)))
    cm = confusion_matrix(y_true, y_pred, labels=present_indices)

    # 1. Save CSV
    csv_p = Path(save_csv)
    csv_p.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_p, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class_name"] + class_names_present)
        for i, name in enumerate(class_names_present):
            writer.writerow([name] + list(cm[i]))
    print(f"[Metrics] Saved confusion matrix CSV to: {csv_p}")

    # 2. Save PNG Heatmap
    png_p = Path(save_png)
    png_p.parent.mkdir(parents=True, exist_ok=True)

    if HAS_MATPLOTLIB:
        plt.figure(figsize=(16, 14))
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title(f"Confusion Matrix ({len(class_names_present)} Test Classes)", fontsize=14, pad=15)
        plt.colorbar(shrink=0.8)

        tick_marks = np.arange(len(class_names_present))
        plt.xticks(tick_marks, tick_marks, rotation=90, fontsize=6)
        plt.yticks(tick_marks, tick_marks, fontsize=6)
        plt.xlabel("Predicted Class Index", fontsize=11)
        plt.ylabel("True Class Index", fontsize=11)
        plt.tight_layout()
        plt.savefig(png_p, dpi=200, bbox_inches="tight")
        plt.close()
    else:
        w, h = 1000, 1000
        canvas = Image.new("RGB", (w, h), color=(255, 255, 255))
        draw = ImageDraw.Draw(canvas)
        draw.rectangle([50, 50, w - 50, h - 50], outline=(100, 100, 100), width=1)
        draw.text((60, 20), f"Confusion Matrix ({len(class_names_present)} Test Classes)", fill=(0, 0, 0))
        canvas.save(png_p, "PNG")

    print(f"[Metrics Plot] Saved confusion matrix PNG heatmap to: {png_p}")

    return cm
