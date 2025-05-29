# shap_app.py

import streamlit as st
import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# Load model
model = joblib.load("rf_fraud_model.pkl")

# Load sample data for SHAP visualization
X_sample = pd.read_csv("data/test_sample.csv")  # Your test input

# Explain model predictions using SHAP
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample)

st.title("🔍 SHAP Explanation for Fraud Detection")

# Select a row to explain
row = st.slider("Select row index to explain", 0, len(X_sample)-1, 0)

st.write("### Input Features")
st.dataframe(X_sample.iloc[[row]])

st.write("### SHAP Force Plot (Matplotlib)")
shap.initjs()
fig = shap.force_plot(explainer.expected_value[1], shap_values[1][row], X_sample.iloc[row], matplotlib=True)
st.pyplot(fig.figure)

st.write("### SHAP Summary Plot (Global Importance)")
plt.figure()
shap.summary_plot(shap_values[1], X_sample)
st.pyplot(plt.gcf())
