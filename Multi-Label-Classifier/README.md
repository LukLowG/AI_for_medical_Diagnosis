# Chest X-ray Multi-Label Classification (DenseNet121)

This notebook tackles the more complex task of detecting multiple diseases per X-ray image using deep learning.

---

## Overview

- Uses **transfer learning** via **DenseNet121** pretrained on ImageNet.
- Classifies **14 possible pathologies** (multi-label problem).
- Leverages TensorFlow 2.x + `tf.data.Dataset` for efficient training.

---

## Dataset

- Source: NIH ChestX-ray14
- Images loaded dynamically via `glob`
- Labels parsed from `Data_Entry_2017.csv`
- Label encoding: Multi-hot vectors (e.g., `[0, 1, 1, 0, ...]`)

---

## Model Architecture

- Base: `DenseNet121` (`include_top=False`, pretrained)
- Head:
  - GlobalAveragePooling
  - Dropout
  - Dense(14, sigmoid)

- **Loss**: Binary crossentropy (multi-label)
- **Metric**: ROC AUC (per class)
- Area Under (ROC) Curve: Measures how well the model seperates positive vs negative examples across all thresholds
- AUC = 1 is perfrect seperation
- AUC ~ 0.5 means the model is guessing randomly
- Each disease is treted as binary classification problem
- Compute an AUC per class
- Average of the AUC per class equals the Overall AUC Score

---

## Training Configuration

- Input size: 224×224
- Batch size: 32
- Optimizer: Adam
- Epochs: 5
- Dataset: Loaded with `tf.data.Dataset` + `AUTOTUNE` for performance

---

## Results

### Training History

- Training + validation accuracy and loss over time

### ROC Curve (per label)

- AUC plotted for all 14 diseases
- Sample labels: Cardiomegaly, Effusion, Pneumonia, Infiltration...

---

## Key Learnings

- Multi-label classification requires:
  - Sigmoid activations
  - Binary loss
  - Evaluation using AUC rather than raw accuracy

- Data pipeline performance matters: switching from a generator to `tf.data.Dataset` improved training time by 3–4×.

---

## 🛠 Next Steps

- Fine-tune DenseNet121 layers
- Add test-time augmentation (TTA)
- Integrate Grad-CAM for visual explanation
