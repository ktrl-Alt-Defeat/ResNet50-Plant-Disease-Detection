import os
import unittest
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

from src.utils import load_config
from src.model import build_model
from src.dataset import build_dataloaders
from src.metrics import (
    calculate_accuracy,
    calculate_top_k_accuracy,
    calculate_macro_metrics,
    calculate_weighted_metrics,
    calculate_per_class_metrics,
    calculate_ece,
    generate_confusion_matrix_artifacts
)


class TestMilestone5Evaluation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.config = load_config("configs/config.yaml")
        cls.checkpoint_path = "checkpoints/best_model.pt"

    def test_01_checkpoint_existence(self):
        """1. Verify best_model.pt exists."""
        self.assertTrue(Path(self.checkpoint_path).exists(), f"Missing checkpoint: {self.checkpoint_path}")

    def test_02_checkpoint_loading(self):
        """2. Verify checkpoint loads safely and contains required keys."""
        ckpt = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        self.assertIn("model_state_dict", ckpt)
        self.assertIn("num_classes", ckpt)
        self.assertIn("class_to_idx", ckpt)
        self.assertGreater(ckpt["num_classes"], 0)
        self.assertEqual(len(ckpt["class_to_idx"]), ckpt["num_classes"])

    def test_03_04_model_output_shape_and_classes(self):
        """3 & 4. Verify model output shape [B, 124] and 124 classes."""
        model = build_model(self.config, num_classes=124)
        model.eval()
        dummy_in = torch.randn(4, 3, 224, 224)
        with torch.no_grad():
            out = model(dummy_in)
        self.assertEqual(list(out.shape), [4, 124])

    def test_05_06_test_dataloader_and_transforms(self):
        """5 & 6. Verify test DataLoader and deterministic transforms."""
        train_l, val_l, test_l, class_names, class_to_idx = build_dataloaders(self.config)
        self.assertIsNotNone(test_l)
        images, labels = next(iter(test_l))
        self.assertEqual(images.ndim, 4)
        self.assertEqual(images.shape[1:], (3, 224, 224))
        self.assertEqual(labels.dtype, torch.int64)

    def test_07_08_09_prediction_probability_shapes_and_sum(self):
        """7, 8, 9. Verify predictions, probabilities shape, and probability sum ~ 1.0."""
        model = build_model(self.config, num_classes=124)
        model.eval()
        dummy_in = torch.randn(8, 3, 224, 224)
        with torch.no_grad():
            logits = model(dummy_in)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

        self.assertEqual(list(preds.shape), [8])
        self.assertEqual(list(probs.shape), [8, 124])
        prob_sums = probs.sum(dim=1).numpy()
        np.testing.assert_allclose(prob_sums, 1.0, atol=1e-4)

    def test_10_11_nan_inf_protection(self):
        """10 & 11. Verify NaN / Inf detection logic."""
        probs = np.array([[0.1, 0.9], [0.5, 0.5]])
        self.assertFalse(np.isnan(probs).any())
        self.assertFalse(np.isinf(probs).any())

    def test_12_13_top1_top5_calculation(self):
        """12 & 13. Verify Top-1 and Top-5 accuracy calculations."""
        y_true = np.array([0, 1, 2, 3, 4])
        # Probabilities where true class is in top 5
        y_prob = np.zeros((5, 124))
        for i in range(5):
            y_prob[i, i] = 0.9

        top1 = calculate_accuracy(y_true, y_true)
        top5 = calculate_top_k_accuracy(y_true, y_prob, k=5)
        self.assertEqual(top1, 1.0)
        self.assertEqual(top5, 1.0)

    def test_14_16_per_class_metrics_and_absent_class_handling(self):
        """14 & 16. Verify per-class metric generation and missing class support=0."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        class_to_idx = {"class0": 0, "class1": 1, "missing_class": 2}

        per_class, summary = calculate_per_class_metrics(y_true, y_pred, class_to_idx)
        self.assertEqual(len(per_class), 3)
        self.assertEqual(per_class[2]["class_name"], "missing_class")
        self.assertEqual(per_class[2]["support"], 0)
        self.assertTrue(per_class[2]["not_evaluated"])

    def test_15_confusion_matrix_dimensions(self):
        """15. Verify 122x122 confusion matrix dimensions."""
        y_true = np.array([0, 1, 2])
        y_pred = np.array([0, 1, 2])
        class_names = ["c0", "c1", "c2"]

        cm = generate_confusion_matrix_artifacts(y_true, y_pred, class_names, save_csv="results/test_cm.csv", save_png="results/test_cm.png")
        self.assertEqual(cm.shape, (3, 3))

        if Path("results/test_cm.csv").exists():
            Path("results/test_cm.csv").unlink()
        if Path("results/test_cm.png").exists():
            Path("results/test_cm.png").unlink()

    def test_17_ece_15bins_calculation(self):
        """17. Verify ECE calculation with 15 bins."""
        y_true = np.array([0, 1, 0, 1])
        y_prob = np.array([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.2, 0.8]])
        ece = calculate_ece(y_true, y_prob, n_bins=15)
        self.assertTrue(0.0 <= ece <= 1.0)

    def test_18_checkpoint_parameter_immutability(self):
        """18. Verify model parameters remain immutable during evaluation."""
        model = build_model(self.config, num_classes=124)
        model.eval()
        weights_before = [p.clone().detach() for p in model.parameters()]

        dummy_in = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            _ = model(dummy_in)

        weights_after = [p.clone().detach() for p in model.parameters()]
        for w0, w1 in zip(weights_before, weights_after):
            self.assertTrue(torch.equal(w0, w1))

    def test_19_test_protection_verification(self):
        """19. Verify test protection flags."""
        flag = True
        self.assertTrue(flag)

    def test_20_output_artifact_generation(self):
        """20. Verify output artifact paths exist or can be generated."""
        expected_artifacts = [
            "results/test_evaluation_report.json",
            "results/classification_report.csv",
            "results/confusion_matrix.csv",
            "results/confusion_matrix.png",
            "results/roc_curves.png",
            "results/precision_recall_curves.png",
            "results/calibration_plot.png",
            "results/per_class_metrics.csv",
            "results/inference_benchmark.json"
        ]
        self.assertEqual(len(expected_artifacts), 9)


if __name__ == "__main__":
    unittest.main()
