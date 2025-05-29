
'''from fastapi import FastAPI, Request
from pydantic import BaseModel
import joblib

app = FastAPI()

# Load model
model = joblib.load("rf_fraud_model.pkl")
# Define input schema
class Transaction(BaseModel):
    distance_from_home: float
    distance_from_last_transaction: float
    ratio_to_median_purchase_price: float
    repeat_retailer: int
    used_chip: int
    used_pin_number: int
    online_order: int
    
@app.get("/")
def read_root():
    return {"message": "Welcome to the API!"}

@app.post("/predict")
def predict(transaction: Transaction):
    input_data = [[
        transaction.distance_from_home,
        transaction.distance_from_last_transaction,
        transaction.ratio_to_median_purchase_price,
        transaction.repeat_retailer,
        transaction.used_chip,
        transaction.used_pin_number,
        transaction.online_order
    ]]
    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0][1]

    return {
        "fraud": bool(prediction),
        "confidence": round(proba, 4)
    }'''

'''from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

# Load your trained model
model = joblib.load("rf_fraud_model.pkl")

# Define input schema
class Transaction(BaseModel):
    distance_from_home: float
    distance_from_last_transaction: float
    ratio_to_median_purchase_price: float
    repeat_retailer: int
    used_chip: int
    used_pin_number: int
    online_order: int

from fastapi.responses import FileResponse

@app.get("/")
def serve_html():
    return FileResponse("static/index.html")

@app.post("/predict")
def predict(transaction: Transaction):
    input_data = [[
        transaction.distance_from_home,
        transaction.distance_from_last_transaction,
        transaction.ratio_to_median_purchase_price,
        transaction.repeat_retailer,
        transaction.used_chip,
        transaction.used_pin_number,
        transaction.online_order
    ]]
    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0][1]

    return {
        "prediction": int(prediction),   # Return int 0 or 1
        "confidence": round(proba, 4)
    }'''
'''from fastapi import FastAPI, HTTPException, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize template engine
templates = Jinja2Templates(directory=".")

# Load the trained model
model = None
try:
    if not os.path.exists("rf_fraud_model.pkl"):
        logger.error("Model file 'rf_fraud_model.pkl' not found. Please run model.py to generate it.")
        raise FileNotFoundError("Model file 'rf_fraud_model.pkl' not found.")
    model = joblib.load("rf_fraud_model.pkl")
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    raise Exception(f"Failed to load model: {str(e)}")

# Define input data model
class Transaction(BaseModel):
    distance_from_home: float
    distance_from_last_transaction: float
    ratio_to_median_purchase_price: float
    repeat_retailer: int
    used_chip: int
    used_pin_number: int
    online_order: int

# Health check endpoint
@app.get("/health")
async def health_check():
    return JSONResponse(content={
        "status": "healthy",
        "model_loaded": model is not None,
        "template_directory": os.path.abspath(".")
    })

# Serve the form
@app.get("/", response_class=HTMLResponse)
async def read_form(request: Request):
    try:
        if not os.path.exists("index.html"):
            logger.error("index.html not found in the current directory.")
            raise HTTPException(status_code=500, detail="Template file 'index.html' not found.")
        return templates.TemplateResponse("index.html", {"request": request})
    except Exception as e:
        logger.error(f"Error rendering template: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error rendering template: {str(e)}")

# Handle prediction
@app.post("/predict")
async def predict(transaction: Transaction):
    try:
        input_data = [[
            transaction.distance_from_home,
            transaction.distance_from_last_transaction,
            transaction.ratio_to_median_purchase_price,
            transaction.repeat_retailer,
            transaction.used_chip,
            transaction.used_pin_number,
            transaction.online_order
        ]]
        prediction = model.predict(input_data)[0]
        confidence = model.predict_proba(input_data)[0][1]
        return JSONResponse(content={
            "prediction": int(prediction),
            "confidence": float(confidence)
        })
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")'''


import streamlit as st
import joblib
import numpy as np
import os
import logging


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the trained model
model = None
if not os.path.exists("rf_fraud_model.pkl"):
    st.error("Model file 'rf_fraud_model.pkl' not found. Please ensure the file is in the working directory.")
else:
    model = joblib.load("rf_fraud_model.pkl")
    logger.info("Model loaded successfully.")

# App title
st.title("Fraud Detection App")

# Health Check Section
with st.expander("🔍 Health Check"):
    st.write("Model loaded:", model is not None)
    st.write("Current directory:", os.getcwd())

# Input form
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
            st.success(f"Prediction: {'Fraud' if prediction == 1 else 'Not Fraud'}")
            st.info(f"Confidence: {confidence:.2f}")
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")

# Your existing input data code...
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

