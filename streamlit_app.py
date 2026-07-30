import streamlit as st

st.title('🎈 App Name')

st.write('Hello world!')
import joblib
import numpy as np
import streamlit as st

# 1. Page Configuration
st.set_page_config(
    page_title="Body Type Prediction App", page_icon="🏋️‍♂️", layout="centered"
)


# 2. Load the trained Joblib Model
@st.cache_resource
def load_model():
    # Loaded model file name changed to KNN.joblib
    model = joblib.load("KNN.joblib")
    return model


try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()

# 3. User Interface Design
st.title("🏋️‍♂️ Body Type Prediction App")
st.write(
    "Enter your physical characteristics below to predict your body type using"
    " our trained K-Nearest Neighbors (KNN) model."
)

st.markdown("---")

# Input widgets for features based on your dataset structure
col1, col2 = st.columns(2)

with col1:
    height_cm = st.number_input(
        "Height (cm)",
        min_value=100.0,
        max_value=250.0,
        value=170.0,
        step=0.5,
        help="Enter your height in centimeters",
    )

with col2:
    weight_kg = st.number_input(
        "Weight (kg)",
        min_value=30.0,
        max_value=200.0,
        value=70.0,
        step=0.5,
        help="Enter your weight in kilograms",
    )

st.markdown("---")

# Define label mapping for numerical outputs to column/category names
# Adjust the mapping order if your model encodes them differently
body_type_mapping = {
    0: "Normal",
    1: "Overweight",
    2: "Underweight"
}

# 4. Prediction Logic
if st.button("Predict Body Type", type="primary", use_container_width=True):
    # Prepare input features as a 2D array matching model expectation
    input_data = np.array([[height_cm, weight_kg]])

    try:
        # Make prediction
        prediction = model.predict(input_data)
        numeric_result = prediction[0]
        
        # Map numerical prediction to category name
        result = body_type_mapping.get(numeric_result, str(numeric_result))

        # Display result with custom styling
        st.success(f"### Predicted Body Type: **{result}**")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

# Footer info
st.markdown("---")
st.caption("Built with Streamlit and scikit-learn.")
