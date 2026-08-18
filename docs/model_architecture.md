# ResNet-50 Model Architecture Specification

This document details the reverse engineered mathematical and architectural specification of the custom PyTorch **ResNet-50** deep learning classifier implemented in [`src/model.py`](file:///d:/resnet%20crop%20detection/src/model.py).

---

## 🏗️ Architecture Design Overview

The model is a custom implementation of standard 50-layer Deep Residual Networks (He et al., 2015). It contains **23.5 Million trainable parameters** and does not use pretrained weights by default (`pretrained: false`), serving as a baseline model.

```
Input Image [B, 3, 224, 224]
        │
        ▼
   [ STEM LAYER ] ──► Conv 7x7 (s=2, p=3) -> BN -> ReLU -> MaxPool 3x3 (s=2, p=1) -> [B, 64, 56, 56]
        │
        ▼
   [ STAGE 1 (Conv2_x) ] ──► 3 Bottleneck Blocks (out: 256 channels)  ──► [B, 256, 56, 56]
        │
        ▼
   [ STAGE 2 (Conv3_x) ] ──► 4 Bottleneck Blocks (out: 512 channels)  ──► [B, 512, 28, 28]
        │
        ▼
   [ STAGE 3 (Conv4_x) ] ──► 6 Bottleneck Blocks (out: 1024 channels) ──► [B, 1024, 14, 14]
        │
        ▼
   [ STAGE 4 (Conv5_x) ] ──► 3 Bottleneck Blocks (out: 2048 channels) ──► [B, 2048, 7, 7]
        │
        ▼
   [ GLOBAL AVG POOL ] ──► AdaptiveAvgPool2d((1, 1))                 ──► [B, 2048, 1, 1]
        │
        ▼
   [ CLASSIFIER HEAD ] ──► Flatten -> Dropout -> Linear(2048 -> N)   ──► [B, Num_Classes]
```

---

## 🧩 Bottleneck Block Structure (`Bottleneck`)

Each residual block utilizes a **1x1 -> 3x3 -> 1x1** bottleneck design with expansion factor $E = 4$:

1. **Conv1 (1x1)**: Reduces channels $C_{in} \to C_{mid}$ (bias=False). Followed by BatchNorm2d + ReLU.
2. **Conv2 (3x3)**: Spatial convolution $C_{mid} \to C_{mid}$ with stride $S \in \{1, 2\}$, padding=1 (bias=False). Followed by BatchNorm2d + ReLU.
3. **Conv3 (1x1)**: Expands channels $C_{mid} \to C_{mid} \times 4$ (bias=False). Followed by BatchNorm2d.
4. **Projection Shortcut / Identity**: If $S \neq 1$ or $C_{in} \neq C_{out}$, a 1x1 Conv ($S$) + BatchNorm downsampling projection is applied to identity.
5. **Residual Addition & Activation**: $Y = \text{ReLU}(\text{Conv3}(X) + \text{Shortcut}(X))$.

---

## 📊 Stage Specifications & Tensor Shapes

| Stage Name | Layer Identifier | Input Tensor Shape | Output Tensor Shape | Bottleneck Blocks | Mid Channels | Out Channels ($C \times 4$) | Stride |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **Stem** | `conv1` + `maxpool` | `[B, 3, 224, 224]` | `[B, 64, 56, 56]` | — | — | 64 | 2 |
| **Stage 1** | `layer1` (Conv2_x) | `[B, 64, 56, 56]` | `[B, 256, 56, 56]` | 3 | 64 | 256 | 1 |
| **Stage 2** | `layer2` (Conv3_x) | `[B, 256, 56, 56]` | `[B, 512, 28, 28]` | 4 | 128 | 512 | 2 |
| **Stage 3** | `layer3` (Conv4_x) | `[B, 512, 28, 28]` | `[B, 1024, 14, 14]` | 6 | 256 | 1024 | 2 |
| **Stage 4** | `layer4` (Conv5_x) | `[B, 1024, 14, 14]` | `[B, 2048, 7, 7]` | 3 | 512 | 2048 | 2 |
| **Pooling** | `avgpool` | `[B, 2048, 7, 7]` | `[B, 2048, 1, 1]` | — | — | 2048 | — |
| **Head** | `fc` | `[B, 2048]` | `[B, Num_Classes]` | — | — | `Num_Classes` | — |

---

## ⚖️ Weight Initialization Strategy (`_initialize_weights`)

- **Convolutions (`nn.Conv2d`)**: Kaiming Normal (He) initialization:
  ```python
  nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
  ```
- **Batch Normalization (`nn.BatchNorm2d`)**: Weights initialized to $1.0$, biases initialized to $0.0$.
- **Zero Residual Initialization (`zero_init_residual=True`)**: The final BatchNorm (`bn3`) of every Bottleneck block is initialized to zero (`gamma = 0`). This allows residual blocks to initially act as identity mappings, accelerating early training convergence.
