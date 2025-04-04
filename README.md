# Chest X-Ray Diagnosis with Deep Learning

This project is a recreation of the "Chest X-Ray Medical Diagnosis with Deep Learning" assignment from the [AI for Medical Diagnosis Specialization](https://www.coursera.org/specializations/ai-for-medicine) by DeepLearning.AI. It demonstrates how to build and evaluate a deep learning model that can detect multiple diseases in chest X-ray images.

---

## Project Objectives

- Load and preprocess the CheXpert chest X-ray dataset. (DONE)
- Build a convolutional neural network (CNN) using TensorFlow and Keras. (DONE)
- Train and evaluate the model using ROC and AUC metrics.
- Interpret model predictions with Grad-CAM visualizations.

---

## Project Structure

chest-xray-classification/
│
├── README.md
│
├── binary-classification/
│   ├── README.md
│   ├── notebook.ipynb
│   └── results/
│
├── multi-label-classification/
    ├── README.md
    ├── notebook.ipynb
    ├── model/
    │   ├── xray_densenet_model.h5
    │   └── best_model.h5
    ├── reports/
        ├── classification_report_1.txt
        ├── classification_report_3.txt
        └── classification_report_5.txt

## Binary Classification

- Goal: detectz whether an X-ray shows signs of pneumonia or is normal
- Model: Custom CNN trained from scratch
- Data: 2-class chest X-ray data (NORMAL vs. PNEUMONIA)
- See the according README.md file wihtin the folder for more information

## Multi-Label-Classification

- Goal: Detect multiple possible pathologies (e.g., Cardiomegaly, Effusion, Pneumonia, etc.) from a single X-ray.
- Model: Transfer learning with DenseNet121 pretrained on ImageNet.
- NIH ChestX-ray14 dataset (14 possible labels per image).
- See the according README.md file wihtin the folder for more information

## Dataset

- In this project the CheXpert datasetis used, available through tensorflow_datasets.
- It includes over 200,000 chest radiographs annotated with 14 common pathologies.
![Figure_1](https://github.com/user-attachments/assets/4f639c98-4c35-4093-baf1-616e6f3456a2)
![Figure_2](https://github.com/user-attachments/assets/b0243789-ece0-495f-b29c-024627691e41)

## Results (Multi-Label Classification)

The multi-label classification model was evaluated at three training stages. Below is a summary of its progression:

| Epochs | Validation AUC | Validation Loss | Precision (micro) | Recall (micro) | F1-score (micro) |
|--------|----------------|------------------|-------------------|----------------|------------------|
| 1      | 0.7133         | 0.1688           | 0.48              | 0.00           | 0.01             |
| 3      | 0.7351         | 0.1679           | 0.46              | 0.01           | 0.02             |
| 5      | 0.7255         | 0.1669           | 0.46              | 0.01           | 0.01             |

- See full classification reports in the "Multi-Label-Classifier" folder for further information

## Interpretation

- **Validation AUC** shows early signal separation, peaking around epoch 3.
- **Recall and F1-score remain near zero** across all labels — the model learned to predict "no disease" due to:
  - Severe class imbalance
  - Too few training epochs
  - Frozen DenseNet layers not yet adapted to medical data

## Future Improvements

- Unfreeze DenseNet121 and fine-tune full network
- Explore Focal Loss for handling class imbalance
- Train for more than 10 epochs with early stopping
- Visualize model decisions with Grad-CAM
- Write up conclusions and insights

## Author

- Lukas Lohr
