import os
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.title("Iris Flower Classifier")
st.write("Adjust the measurements below and click **Predict** to classify the flower.")

col1, col2 = st.columns(2)
with col1:
    sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.1, 0.1)
    sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.5, 0.1)
with col2:
    petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.4, 0.1)
    petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2, 0.1)

if st.button("Predict", type="primary"):
    payload = {
        "sepal_length": sepal_length,
        "sepal_width": sepal_width,
        "petal_length": petal_length,
        "petal_width": petal_width,
    }
    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=5)
        response.raise_for_status()
        result = response.json()
        st.success(f"Prediction: **{result['class_name'].capitalize()}**")
        st.info(f"Confidence: {result['confidence'] * 100:.1f}%")
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to the API. Make sure the FastAPI service is running.")
    except Exception as e:
        st.error(f"Error: {e}")
