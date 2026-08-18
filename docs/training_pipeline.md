# Training & Optimization Engine Documentation

This document provides a reverse engineered breakdown of the training lifecycle, optimization algorithms, mixed precision mechanics, learning rate scheduling, and early stopping implemented in [`src/train.py`](file:///d:/resnet%20crop%20detection/src/train.py) and [`src/utils.py`](file:///d:/resnet%20crop%20detection/src/utils.py).

---

## ⚙️ Optimization & Hyperparameter Configurations

Hyperparameters are managed via [`configs/config.yaml`](file:///d:/resnet%20crop%20detection/configs/config.yaml):

| Hyperparameter | Configuration Value | Target Function / PyTorch Class |
| :--- | :--- | :--- |
| **Optimizer** | `AdamW` | `torch.optim.AdamW` |
| **Base Learning Rate** | `3e-4` (`0.0003`) | `lr=0.0003` |
| **Weight Decay** | `1e-4` (`0.0001`) | `weight_decay=0.0001` |
| **Learning Rate Scheduler**| `CosineAnnealingLR` | `torch.optim.lr_scheduler.CosineAnnealingLR` |
| **Minimum Learning Rate** | `1e-6` (`0.000001`) | `eta_min=0.000001` |
| **Loss Function** | `CrossEntropyLoss` | `torch.nn.CrossEntropyLoss` |
| **Label Smoothing** | `0.0` (Standard) | `label_smoothing=0.0` |
| **Mixed Precision** | `AMP Enabled` | `torch.cuda.amp.autocast()`, `GradScaler()` |
| **Gradient Clipping** | `max_norm = 1.0` | `torch.nn.utils.clip_grad_norm_()` |
| **Batch Size** | `32` | DataLoader `batch_size=32` |
| **Epochs** | `50` | Outer loop iterations |
| **Early Stopping** | `Patience = 10` | Monitor `val_loss` (Mode: `min`) |

---

## 🔄 Training Epoch Loop Workflow (`train_one_epoch`)

```
For each batch (inputs, targets) in train_loader:
    │
    ├── 1. Move tensors to active device (CUDA GPU / CPU)
    ├── 2. Zero gradients: optimizer.zero_grad(set_to_none=True)
    ├── 3. Forward pass with AMP:
    │      with torch.cuda.amp.autocast(enabled=mixed_precision):
    │          logits = model(inputs)
    │          loss = criterion(logits, targets)
    ├── 4. Backward pass with GradScaler:
    │      scaler.scale(loss).backward()
    ├── 5. Unscale & Clip Gradients:
    │      scaler.unscale_(optimizer)
    │      torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    ├── 6. Optimizer & Scaler Step:
    │      scaler.step(optimizer)
    │      scaler.update()
    └── 7. Accumulate batch loss & compute top-1 accuracy
```

---

## 📈 Learning Rate Annealing & Early Stopping

### Cosine Annealing Learning Rate Scheduler
At the end of each epoch, the scheduler updates the learning rate following cosine decay:
$$\eta_t = \eta_{\text{min}} + \frac{1}{2}(\eta_{\text{base}} - \eta_{\text{min}})\left(1 + \cos\left(\frac{t}{T_{\text{max}}}\pi\right)\right)$$
Where $T_{\text{max}} = \text{total\_epochs} = 50$, $\eta_{\text{base}} = 3\times 10^{-4}$, $\eta_{\text{min}} = 10^{-6}$.

### Early Stopping Engine (`EarlyStopping`)
- **Monitored Metric**: Validation Loss (`val_loss`).
- **Patience Counter**: Incremented when `val_loss` fails to improve by at least `min_delta = 1e-4`.
- **Termination**: If patience counter reaches `10`, training terminates early to prevent overfitting.
- **Checkpoint Persistence**: Whenever validation loss reaches a new global minimum, `checkpoints/best_model.pt` is updated with full model state dict, optimizer state, epoch, and class mapping metadata.
