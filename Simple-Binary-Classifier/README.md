# Simple Chest X-ray Binary Classification

This notebook demonstrates a basic CNN approach to classify chest X-ray images as either:

- **NORMAL**
- **PNEUMONIA**

---

## Overview

- Uses grayscale chest X-ray images.
- Labels are derived from folder names (`NORMAL/`, `PNEUMONIA/`).
- Trains a simple Convolutional Neural Network from scratch.

---

## Dataset

- Source: Chest X-ray Pneumonia dataset
- Format: Two folders, each representing one class.
- Images are resized to 100×100 for faster training.

---

## Model Architecture

```text
Conv2D(64) + ReLU → MaxPooling
Conv2D(64) + ReLU → MaxPooling
Flatten → Dense(64) → Dense(1, sigmoid)
