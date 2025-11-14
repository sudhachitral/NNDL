import streamlit as st
import numpy as np
import joblib

# Load Model
model = joblib.load("model.pkl")   # change file name if needed

st.title("🔬 Breast Cancer Classification")
st.write("Enter the features to predict whether the tumor is Benign or Malignant.")

# Input fields (30 features)
feature_inputs = []
feature_names = [
    "mean radius", "mean texture", "mean perimeter", "mean area", "mean smoothness",
    "mean compactness", "mean concavity", "mean concave points", "mean symmetry",
    "mean fractal dimension", "radius error", "texture error", "perimeter error",
    "area error", "smoothness error", "compactness error", "concavity error",
    "concave points error", "symmetry error", "fractal dimension error",
    "worst radius", "worst texture", "worst perimeter", "worst area",
    "worst smoothness", "worst compactness", "worst concavity",
    "worst concave points", "worst symmetry", "worst fractal dimension"
]

for name in feature_names:
    value = st.number_input(name, format="%.4f")
    feature_inputs.append(value)

if st.button("Predict"):
    input_data = np.array(feature_inputs).reshape(1, -1)
    prediction = model.predict(input_data)[0]

    if prediction == 0:
        st.success("✔ The tumor is **Malignant**.")
    else:
        st.success("✔ The tumor is **Benign**.")
