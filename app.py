import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Load model and scaler
model = load_model("final_model.h5")
scaler = joblib.load("scaler.joblib")

st.title("Breast Cancer Classification App")
st.write("Enter the 30 feature values from the dataset to classify the tumor.")

# Input fields for 30 features
feature_names = [
    'mean_radius', 'mean_texture', 'mean_perimeter', 'mean_area', 'mean_smoothness',
    'mean_compactness', 'mean_concavity', 'mean_concave_points', 'mean_symmetry', 'mean_fractal_dimension',
    'radius_error', 'texture_error', 'perimeter_error', 'area_error', 'smoothness_error',
    'compactness_error', 'concavity_error', 'concave_points_error', 'symmetry_error', 'fractal_dimension_error',
    'worst_radius', 'worst_texture', 'worst_perimeter', 'worst_area', 'worst_smoothness',
    'worst_compactness', 'worst_concavity', 'worst_concave_points', 'worst_symmetry', 'worst_fractal_dimension'
]

inputs = []

for name in feature_names:
    value = st.number_input(f"{name.replace('_',' ').title()}", value=0.0)
    inputs.append(value)

if st.button("Predict"):
    X = np.array([inputs])
    X_scaled = scaler.transform(X)
    prediction = model.predict(X_scaled)[0][0]

    st.write("### 🔍 Prediction Result")

    if prediction > 0.5:
        st.success("🟢 **Benign Tumor**")
    else:
        st.error("🔴 **Malignant Tumor**")

