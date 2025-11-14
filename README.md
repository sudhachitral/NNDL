# Breast Cancer Classification using Neural Networks

This project classifies breast cancer tumors as **malignant** or **benign** using a Neural Network built with TensorFlow/Keras. It uses the **Breast Cancer Wisconsin (Diagnostic) dataset** which contains **569 samples** and **30 numerical features** extracted from digitized images of breast tissue.

## Project Overview
The main goal of this project is to demonstrate how a Neural Network can be used for medical data classification. The model evaluates tumor types and provides clear performance metrics to assess its effectiveness.

### Features:
- **Neural Network architecture** with Dropout and Batch Normalization for better generalization
- **Evaluation metrics:** Accuracy, ROC AUC, Confusion Matrix
- **Visualization:** Loss and Accuracy curves, ROC Curve
- **Model persistence:** Saves trained model (`final_model.h5`) and scaler (`scaler.joblib`) for future predictions
- **Handles class imbalance** using class weights

Dataset
Source: UCI Machine Learning Repository
https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
Also available via: Built-in sklearn.datasets.load_breast_cancer()
Samples: 569
Features: 30 numeric features
Target Classes:
0 = Malignant
1 = Benign
  

## Requirements
- Python 3.x  
- TensorFlow  
- scikit-learn  
- pandas  
- numpy  
- matplotlib  
- seaborn  
- joblib  

You can install the required packages using:
```bash
pip install -r requirements.txt
