import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Page configuration
st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="centered")

st.title("🏠 California House Price Predictor")
st.write("Adjust the sliders on the sidebar to input property features and predict the estimated market price.")

# Load Model
@st.cache_resource
def load_model():
    return joblib.load('house_price_model.pkl')

try:
    artifacts = load_model()
    model = artifacts['model']
    scaler = artifacts['scaler']

    # User Input Sidebar
    st.sidebar.header("Property Features")
    
    med_inc = st.sidebar.slider("Median Income ($10k units)", 0.5, 15.0, 3.5, step=0.1)
    house_age = st.sidebar.slider("House Age (Years)", 1, 52, 25)
    ave_rooms = st.sidebar.slider("Average Rooms", 1.0, 10.0, 5.0, step=0.1)
    ave_bedrms = st.sidebar.slider("Average Bedrooms", 0.5, 5.0, 1.0, step=0.1)
    population = st.sidebar.slider("Block Population", 3, 3000, 1000)
    ave_occup = st.sidebar.slider("Average Household Occupancy", 1.0, 6.0, 3.0, step=0.1)
    latitude = st.sidebar.slider("Latitude", 32.5, 42.0, 36.0, step=0.1)
    longitude = st.sidebar.slider("Longitude", -124.5, -114.0, -119.0, step=0.1)

    # Convert Inputs to Array
    input_data = pd.DataFrame([[
        med_inc, house_age, ave_rooms, ave_bedrms, population, ave_occup, latitude, longitude
    ]], columns=artifacts['feature_names'])

    # Scale Inputs
    scaled_input = scaler.transform(input_data)

    # Prediction Button
    if st.button("Predict Price", type="primary"):
        prediction = model.predict(scaled_input)[0]
        estimated_price = prediction * 100,000  # Dataset target is in $100k units
        
        st.success(f"### Estimated Price: **${estimated_price:,.2f}**")
        
        # Display Feature Summary
        st.subheader("Submitted Parameters")
        st.dataframe(input_data)

except FileNotFoundError:
    st.error("Model file 'house_price_model.pkl' not found! Make sure to train and export the model first.")
