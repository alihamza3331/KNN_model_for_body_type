import streamlit as st

st.title('🎈KNN model for Predicting the Body Type ')

st.write('Build By Ali Hamza and thanks to Sir Zafer for teching us how to make app like this')
import joblib
import numpy as np
import streamlit as st

# 1. Page Configuration
st.set_page_config(
    page_title="Body Type Prediction App", page_icon="🏋️‍♂️", layout="centered"
)


# 2. Load the trained Joblib Model and Scaler
@st.cache_resource
def load_artifacts():
    model = joblib.load("KNN.joblib")
    # You MUST also save and load your scaler if you used one during training!
    scaler = joblib.load("scaler.joblib") 
    return model, scaler


try:
    model, scaler = load_artifacts()
except Exception as e:
    st.error(f"Error loading model or scaler: {e}")
    st.info("Make sure 'KNN.joblib' and 'scaler.joblib' are in your project folder.")
    st.stop()

# 3. User Interface Design
st.title("🏋️‍♂️ Body Type Prediction App")
st.write(
    "Enter your physical characteristics below to predict your body type using"
    " our trained K-Nearest Neighbors (KNN) model."
)

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    height_cm = st.number_input(
        "Height (cm)",
        min_value=100.0,
        max_value=250.0,
        value=170.0,
        step=0.5,
    )

with col2:
    weight_kg = st.number_input(
        "Weight (kg)",
        min_value=30.0,
        max_value=200.0,
        value=70.0,
        step=0.5,
    )

st.markdown("---")

body_type_mapping = {
    0: "Normal",
    1: "Overweight",
    2: "Underweight"
}

# 4. Prediction Logic
if st.button("Predict Body Type", type="primary", use_container_width=True):
    # Prepare input features as a 2D array
    input_data = np.array([[height_cm, weight_kg]])

    try:
        # CRITICAL: Scale the input data just like you did during training
        scaled_input_data = scaler.transform(input_data)

        # Make prediction using the scaled data
        prediction = model.predict(scaled_input_data)
        numeric_result = prediction[0]
        
        # Map numerical prediction to category name
        result = body_type_mapping.get(numeric_result, str(numeric_result))

        st.success(f"### Predicted Body Type: **{result}**")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

st.markdown("---")
st.caption("Built by Ali Hamza thanks sir zafer your a Gem.")
