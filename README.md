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

chest-xray-diagnosis/\
├── README.md # This file\
├── Old Versions # Python dependencies\
├── Simple-Binary-Classifier \
├── Multi-Label-Classifier \
├── results \
├── archive/ # local data storage\
└── images/ # Visualizations and result figures\

## Dataset

- In this project the CheXpert datasetis used, available through tensorflow_datasets.
- It includes over 200,000 chest radiographs annotated with 14 common pathologies.
![Figure_1](https://github.com/user-attachments/assets/4f639c98-4c35-4093-baf1-616e6f3456a2)
![Figure_2](https://github.com/user-attachments/assets/b0243789-ece0-495f-b29c-024627691e41)
## TODOs

- Load and preview dataset
- Build and train a CNN model
- Evaluate using AUC and ROC
- Visualize model decisions with Grad-CAM
- Write up conclusions and insights

## Author

- Lukas Lohr

