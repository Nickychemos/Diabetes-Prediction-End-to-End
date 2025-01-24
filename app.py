# Importing necessary libraries
import streamlit as st
import numpy as np
import joblib

@st.cache_resource
def load_model():
    """Load the trained model."""
    return joblib.load('Model_gb1.pkl')

st.cache_resource.clear()  # Ensures changes are applied immediately.

# App title and description
st.title("Diabetes Prediction Application")
st.subheader("This application helps predict whether an individual has diabetes based on the provided metrics, thus providing insights for early detection and preventive healthcare strategies.")

# Load the model
model = load_model()

if model:
    st.header("Please Enter The Following Details")

# User input fields
heart_disease = st.selectbox("Has heart disease?", options=[(0, "No"), (1, "Yes")], format_func=lambda x: x[1])
heart_disease_value = heart_disease[0]  # Extract the numeric value

hypertension = st.selectbox("Has hypertension?", options=[(0, "No"), (1, "Yes")], format_func=lambda x: x[1])
hypertension_value = hypertension[0]  # Extract the numeric value

age = st.number_input("Age", min_value=0, max_value=100, value=0)
bmi = st.number_input("BMI", min_value=10.0, max_value=100.0, value=10.0, step=0.1)
blood_glucose_level = st.number_input("Blood Glucose Level", min_value=0.0, max_value=300.0, value=0.0, step=0.1)
HbA1c_level = st.number_input("HbA1c Level", min_value=0.0, max_value=10.0, value=0.0, step=0.1)

# Prepare input data
input_data = np.array([age, hypertension_value, heart_disease_value, bmi, HbA1c_level, blood_glucose_level])

# Prediction button
if st.button("Predict Diabetes Status", key="predict_button"):
    prediction = model.predict(input_data.reshape(1, -1))  # Ensure the input is in the correct shape
    diabetes_status = "Yes" if prediction[0] == 1 else "No"
    st.subheader(f"Predicted Diabetes Status: {diabetes_status}")
