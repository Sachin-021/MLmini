import streamlit as st
import joblib
import numpy as np
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = None
if not os.path.exists("rf_fraud_model.pkl"):
    st.error("Model file 'rf_fraud_model.pkl' not found. Please ensure the file is in the working directory.")
else:
    model = joblib.load("rf_fraud_model.pkl")
    logger.info("Model loaded successfully.")

st.title("Credit Card Fraud Detection")

st.header("Enter Transaction Details")

distance_from_home = st.number_input("Distance from Home", min_value=0.0, value=5.0)
distance_from_last_transaction = st.number_input("Distance from Last Transaction", min_value=0.0, value=3.0)
ratio_to_median_purchase_price = st.number_input("Ratio to Median Purchase Price", min_value=0.0, value=1.0)
repeat_retailer = st.selectbox("Repeat Retailer", [0, 1])
used_chip = st.selectbox("Used Chip", [0, 1])
used_pin_number = st.selectbox("Used PIN Number", [0, 1])
online_order = st.selectbox("Online Order", [0, 1])

if st.button("Predict Fraud"):
    if model is None:
        st.error("Model not loaded.")
    else:
        input_data = [[
            distance_from_home,
            distance_from_last_transaction,
            ratio_to_median_purchase_price,
            repeat_retailer,
            used_chip,
            used_pin_number,
            online_order
        ]]
        try:
            prediction = model.predict(input_data)[0]
            confidence = model.predict_proba(input_data)[0][1]
           
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")

input_data = [[
    distance_from_home,
    distance_from_last_transaction,
    ratio_to_median_purchase_price,
    repeat_retailer,
    used_chip,
    used_pin_number,
    online_order
]]

# Prediction
prediction = model.predict(input_data)[0]
confidence = model.predict_proba(input_data)[0][1]

# Show result
st.success(f"🎯 Prediction: {'Fraud (1)' if prediction == 1 else 'Not Fraud (0)'}")
st.info(f"📊 Confidence: {confidence:.4%}")
