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
├── train.py # Python script for training
├── README.md # This file\
├── requirements.txt # Python dependencies\
├── archive/ # local data storage\
└── images/ # Visualizations and result figures\

## Dataset

- In this project the CheXpert datasetis used, available through tensorflow_datasets.
- It includes over 200,000 chest radiographs annotated with 14 common pathologies.

## TODOs

- Load and preview dataset
- Build and train a CNN model
- Evaluate using AUC and ROC
- Visualize model decisions with Grad-CAM
- Write up conclusions and insights

## Author

- Lukas Lohr
