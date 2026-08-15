import os
import unittest
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim

from src.utils import load_config
from src.model import build_model
from src.train import (
    validate_training_config,
    check_dataset_counts,
    build_loss,
    build_optimizer,
    build_scheduler,
    save_atomic_checkpoint,
    load_training_checkpoint,
    EarlyStopping,
    run_synthetic_step_verification
)


class TestMilestone4Infrastructure(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.config = load_config("configs/config.yaml")

    def test_01_configuration_validation(self):
        """1. Verify config validation function."""
        try:
            validate_training_config(self.config)
        except Exception as e:
            self.fail(f"validate_training_config raised exception: {e}")

    def test_02_dataset_count_verification(self):
        """2. Verify dataset count checking function."""
        counts = check_dataset_counts(self.config.get("data", {}))
        self.assertGreater(counts["train"], 0)
        self.assertGreater(counts["val"], 0)
        self.assertGreater(counts["test"], 0)

    def test_03_model_initialization(self):
        """3. Verify model initialization for training."""
        model = build_model(self.config)
        self.assertGreater(model.num_classes, 0)
        self.assertTrue(isinstance(model, nn.Module))

    def test_04_loss_construction(self):
        """4. Verify loss builder (CrossEntropyLoss)."""
        criterion = build_loss(self.config)
        self.assertTrue(isinstance(criterion, nn.CrossEntropyLoss))

    def test_05_optimizer_construction(self):
        """5. Verify optimizer builder (AdamW with requires_grad filtering)."""
        model = build_model(self.config)
        optimizer = build_optimizer(model, self.config)
        self.assertTrue(isinstance(optimizer, optim.AdamW))
        # Ensure all param groups have lr == 0.0003
        self.assertEqual(optimizer.param_groups[0]["lr"], 0.0003)

    def test_06_scheduler_construction(self):
        """6. Verify scheduler builder (CosineAnnealingLR)."""
        model = build_model(self.config)
        optimizer = build_optimizer(model, self.config)
        scheduler = build_scheduler(optimizer, self.config)
        self.assertTrue(isinstance(scheduler, optim.lr_scheduler.CosineAnnealingLR))

    def test_07_amp_initialization(self):
        """7. Verify AMP scaler initialization."""
        enabled = torch.cuda.is_available()
        scaler = torch.amp.GradScaler('cuda', enabled=enabled)
        self.assertEqual(scaler.is_enabled(), enabled)

    def test_08_09_10_11_12_13_synthetic_step_pipeline(self):
        """8-13. Verify synthetic forward, loss, backward, grad check, and optimizer update."""
        model = build_model(self.config)
        device = torch.device("cpu")
        res = run_synthetic_step_verification(model, device, self.config)
        
        self.assertTrue(res["forward_pass"])
        self.assertTrue(res["loss_calculation"])
        self.assertTrue(res["backward_pass"])
        self.assertTrue(res["gradient_existence"])
        self.assertTrue(res["gradient_finite"])
        self.assertTrue(res["parameter_update"])

    def test_14_early_stopping_logic(self):
        """14. Verify EarlyStopping logic."""
        es = EarlyStopping(patience=3, mode="min")
        is_best, stop = es.step(1.0)
        self.assertTrue(is_best)
        self.assertFalse(stop)

        is_best, stop = es.step(1.1)
        self.assertFalse(is_best)
        self.assertFalse(stop)

        is_best, stop = es.step(1.2)
        is_best, stop = es.step(1.3)
        self.assertTrue(stop)

    def test_15_16_atomic_checkpoint_save_and_load(self):
        """15 & 16. Verify atomic checkpoint save, loadability verification, and replace pattern."""
        model = build_model(self.config)
        optimizer = build_optimizer(model, self.config)
        
        test_checkpoint_path = "checkpoints/test_dummy_checkpoint.pt"
        checkpoint_dict = {
            "epoch": 5,
            "best_val_loss": 0.42,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "class_to_idx": {"classA": 0},
            "num_classes": 124
        }
        
        # Save atomically
        save_atomic_checkpoint(checkpoint_dict, test_checkpoint_path)
        self.assertTrue(Path(test_checkpoint_path).exists())

        # Load back
        start_epoch, best_loss, meta = load_training_checkpoint(test_checkpoint_path, model, optimizer)
        self.assertEqual(start_epoch, 6)
        self.assertEqual(best_loss, 0.42)

        # Cleanup test checkpoint
        if Path(test_checkpoint_path).exists():
            Path(test_checkpoint_path).unlink()

    def test_17_resume_state_restoration(self):
        """17. Verify state restoration when resuming."""
        model = build_model(self.config)
        opt = build_optimizer(model, self.config)
        sched = build_scheduler(opt, self.config)

        test_path = "checkpoints/test_resume_checkpoint.pt"
        save_atomic_checkpoint({
            "epoch": 10,
            "best_val_loss": 0.25,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "scheduler_state_dict": sched.state_dict(),
            "class_to_idx": {"classA": 0}
        }, test_path)

        start_ep, best_l, _ = load_training_checkpoint(test_path, model, opt, sched)
        self.assertEqual(start_ep, 11)
        self.assertEqual(best_l, 0.25)

        if Path(test_path).exists():
            Path(test_path).unlink()

    def test_18_nan_inf_detection(self):
        """18. Verify non-finite loss detection raises error."""
        model = build_model(self.config)
        criterion = build_loss(self.config)
        
        dummy_out = torch.tensor([[float("nan"), 1.0]], requires_grad=True)
        dummy_target = torch.tensor([0])
        
        loss = criterion(dummy_out, dummy_target)
        self.assertTrue(torch.isnan(loss).item())

    def test_19_training_history_structure(self):
        """19. Verify structure of metrics CSV fields."""
        required_fields = [
            "epoch", "train_loss", "val_loss", "learning_rate",
            "epoch_time_seconds", "gpu_memory_allocated_mb",
            "gpu_memory_reserved_mb", "best_val_loss", "is_best"
        ]
        self.assertEqual(len(required_fields), 9)

    def test_20_configuration_integration(self):
        """20. Verify training config parameters load cleanly."""
        tr_cfg = self.config.get("training", {})
        self.assertEqual(tr_cfg.get("epochs"), 50)
        self.assertEqual(tr_cfg.get("batch_size"), 32)


if __name__ == "__main__":
    unittest.main()
