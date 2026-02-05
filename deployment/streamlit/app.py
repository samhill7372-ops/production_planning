"""
Yield Prediction API Testing UI

Simple Streamlit app for testing the FastAPI yield prediction endpoints.
Connects to API via environment variable API_URL.
"""

import streamlit as st
import requests
import os
import pandas as pd
import plotly.express as px

# API URL from environment variable
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Yield Prediction API Tester",
    page_icon="🧪",
    layout="wide"
)

st.title("🧪 Yield Prediction API Tester")
st.caption(f"Connected to: `{API_URL}`")


# Health Check
def check_health():
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.json()
    except Exception as e:
        return {"error": str(e)}


# Sidebar - API Status
with st.sidebar:
    st.header("API Status")
    if st.button("Check Health"):
        health = check_health()
        if "error" in health:
            st.error(f"API Error: {health['error']}")
        else:
            st.success("API is healthy!")
            st.json(health)

    st.divider()
    st.header("Model Info")
    try:
        info = requests.get(f"{API_URL}/model/info", timeout=5).json()
        st.write(f"**Model Year:** {info.get('model_year', 'N/A')}")
        st.write(f"**Features:** {info.get('num_features', 'N/A')}")
    except:
        st.warning("Unable to fetch model info")


# Main tabs
tab1, tab2, tab3 = st.tabs(["Forward Prediction", "Reverse Prediction", "Raw API"])

with tab1:
    st.header("Forward Prediction")
    st.write("Predict yield given input parameters")

    col1, col2 = st.columns(2)

    with col1:
        ks_grade = st.selectbox("KS Grade", ["KS-A", "KS-B", "KS-C", "KS-D", "KS-E"])
        kd_output = st.selectbox("KD Output", ["KD-01", "KD-02", "KD-03", "KD-04", "KD-05"])
        thickness = st.number_input("Thickness (mm)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
        width = st.number_input("Width (mm)", min_value=10.0, max_value=2000.0, value=100.0, step=10.0)

    with col2:
        input_bf = st.number_input("Input BF (kg)", min_value=100.0, max_value=100000.0, value=1000.0, step=100.0)

    if st.button("Predict Yield", type="primary"):
        payload = {
            "ks_grade": ks_grade,
            "kd_output": kd_output,
            "thickness": thickness,
            "width": width,
            "input_bf": input_bf
        }

        try:
            response = requests.post(f"{API_URL}/predict/forward", json=payload, timeout=10)
            result = response.json()

            if response.status_code == 200:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Predicted Yield", f"{result.get('predicted_yield', 0):.2%}")
                with col2:
                    st.metric("Expected Output", f"{result.get('expected_output_bf', 0):.1f} kg")
                with col3:
                    st.metric("Confidence", result.get('confidence', 'N/A'))

                with st.expander("Full Response"):
                    st.json(result)
            else:
                st.error(f"Error: {result}")
        except Exception as e:
            st.error(f"Request failed: {e}")


with tab2:
    st.header("Reverse Prediction")
    st.write("Calculate required input given desired output")

    col1, col2 = st.columns(2)

    with col1:
        ks_grade_r = st.selectbox("KS Grade", ["KS-A", "KS-B", "KS-C", "KS-D", "KS-E"], key="ks_r")
        kd_output_r = st.selectbox("KD Output", ["KD-01", "KD-02", "KD-03", "KD-04", "KD-05"], key="kd_r")
        thickness_r = st.number_input("Thickness (mm)", min_value=0.1, max_value=10.0, value=1.0, step=0.1, key="thick_r")
        width_r = st.number_input("Width (mm)", min_value=10.0, max_value=2000.0, value=100.0, step=10.0, key="width_r")

    with col2:
        target_output = st.number_input("Target Output BF (kg)", min_value=100.0, max_value=100000.0, value=900.0, step=100.0)

    if st.button("Calculate Required Input", type="primary"):
        payload = {
            "ks_grade": ks_grade_r,
            "kd_output": kd_output_r,
            "thickness": thickness_r,
            "width": width_r,
            "target_output_bf": target_output
        }

        try:
            response = requests.post(f"{API_URL}/predict/reverse", json=payload, timeout=10)
            result = response.json()

            if response.status_code == 200:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Required Input", f"{result.get('required_input_bf', 0):.1f} kg")
                with col2:
                    st.metric("Expected Yield", f"{result.get('expected_yield', 0):.2%}")
                with col3:
                    st.metric("Waste Estimate", f"{result.get('estimated_waste', 0):.1f} kg")

                with st.expander("Full Response"):
                    st.json(result)
            else:
                st.error(f"Error: {result}")
        except Exception as e:
            st.error(f"Request failed: {e}")


with tab3:
    st.header("Raw API Tester")
    st.write("Send custom requests to the API")

    method = st.selectbox("Method", ["GET", "POST"])
    endpoint = st.text_input("Endpoint", value="/health")

    if method == "POST":
        body = st.text_area("JSON Body", value='{"key": "value"}', height=200)

    if st.button("Send Request"):
        try:
            url = f"{API_URL}{endpoint}"
            if method == "GET":
                response = requests.get(url, timeout=10)
            else:
                import json
                response = requests.post(url, json=json.loads(body), timeout=10)

            st.write(f"**Status Code:** {response.status_code}")
            st.json(response.json())
        except Exception as e:
            st.error(f"Request failed: {e}")
