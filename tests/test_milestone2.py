import unittest
import json
import csv
from pathlib import Path
import torch

from src.utils import load_config
from src.dataset_validation import (
    validate_class_alignment,
    check_split_integrity,
    detect_duplicates_and_leakage,
    calculate_distribution_stats,
    infer_and_save_class_mapping
)
from src.dataset import (
    infer_num_classes,
    get_class_mapping,
    get_transforms,
    create_datasets,
    build_dataloaders,
    verify_dataloader_batches
)


class TestMilestone2Pipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.config = load_config("configs/config.yaml")

    def test_01_dataset_directories(self):
        """1. Verify dataset split directories exist."""
        data_cfg = self.config.get("data", {})
        self.assertTrue(Path(data_cfg["train_dir"]).exists(), "Train dir missing")
        self.assertTrue(Path(data_cfg["val_dir"]).exists(), "Val dir missing")
        self.assertTrue(Path(data_cfg["test_dir"]).exists(), "Test dir missing")

    def test_02_class_inference(self):
        """2. Verify dynamic class inference from train directory."""
        num_classes, class_names = infer_num_classes("data/train")
        self.assertGreater(num_classes, 0, "No classes inferred")
        self.assertEqual(len(class_names), num_classes, "Class count mismatch")

    def test_03_class_consistency(self):
        """3. Verify class alignment detection across splits."""
        report = validate_class_alignment("data/train", "data/val", "data/test")
        self.assertIn("train_classes", report)
        self.assertIn("aligned", report)

    def test_04_corrupted_image_detection(self):
        """4. Verify corruption scanner returns zero corrupted images on dataset."""
        train_report = check_split_integrity("train", "data/train")
        self.assertEqual(len(train_report["corrupted_files"]), 0, "Corrupted files found in train")

    def test_05_06_duplicate_and_leakage_detection(self):
        """5 & 6. Verify duplicate and leakage scanner."""
        dup_report = detect_duplicates_and_leakage({
            "train": "data/train",
            "val": "data/val",
            "test": "data/test"
        })
        self.assertFalse(dup_report["has_leakage"], "Data leakage detected")
        self.assertEqual(dup_report["cross_split_leakage_count"], 0)

    def test_07_class_distribution(self):
        """7. Verify class distribution stats calculation."""
        class_counts = {"classA": 10, "classB": 20, "classC": 30}
        stats = calculate_distribution_stats(class_counts)
        self.assertEqual(stats["total_images"], 60)
        self.assertEqual(stats["min_class_size"], 10)
        self.assertEqual(stats["max_class_size"], 30)
        self.assertEqual(stats["imbalance_ratio"], 3.0)

    def test_08_transform_creation(self):
        """8. Verify transform pipeline creation."""
        train_tf, val_tf = get_transforms(self.config)
        self.assertIsNotNone(train_tf)
        self.assertIsNotNone(val_tf)

    def test_09_10_dataloader_creation_and_batch_loading(self):
        """9 & 10. Verify DataLoaders construction and batch fetching."""
        train_loader, val_loader, test_loader, class_names, class_to_idx = build_dataloaders(self.config)
        self.assertIsNotNone(train_loader)
        self.assertIsNotNone(val_loader)
        self.assertIsNotNone(test_loader)

        images, labels = next(iter(train_loader))
        batch_size = self.config["data"]["batch_size"]
        self.assertEqual(images.shape, torch.Size([batch_size, 3, 224, 224]))
        self.assertEqual(labels.shape, torch.Size([batch_size]))

    def test_11_deterministic_class_mapping(self):
        """11. Verify deterministic class_to_idx mapping and JSON export."""
        mapping = get_class_mapping("data/train", save_path="results/class_to_idx.json")
        self.assertGreater(len(mapping), 0)
        first_class = sorted(list(mapping.keys()))[0]
        self.assertEqual(mapping[first_class], 0)

    def test_12_configuration_integration(self):
        """12. Verify config parameter integration."""
        data_cfg = self.config.get("data", {})
        self.assertEqual(data_cfg["img_size"], 224)
        self.assertFalse(data_cfg.get("drop_last", True), "drop_last should be False")


if __name__ == "__main__":
    unittest.main()
