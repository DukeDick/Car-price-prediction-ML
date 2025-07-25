import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor  # Required if model uses this

# 🎯 Load the trained Random Forest model
with open("random_forest_regression_model.pkl", "rb") as f:
    model = pickle.load(f)

# 🔑 Feature order (must match training data)
features = [
    'Present_Price', 'Driven_kms', 'Owner', 'no_year',
    'Fuel_Type_CNG', 'Fuel_Type_Diesel', 'Fuel_Type_Petrol',
    'Selling_type_Dealer', 'Selling_type_Individual',
    'Transmission_Automatic', 'Transmission_Manual'
]

# 🧠 Streamlit UI
st.title("🚗 Car Price Predictor")
st.markdown("Estimate your **used car's selling price** in **USD** using a trained ML model.")

# 🚙 User Inputs
usd_price = st.number_input("Original Price When New (USD)", min_value=0.0, step=100.0)
present_price = (usd_price * 83) / 100000  # Convert USD → lakhs INR
st.caption("_Internally converted to ₹ lakhs (1 USD ≈ ₹83)_")

driven_kms = st.number_input("Driven Kilometers", min_value=0)
owner = st.selectbox("Number of Previous Owners", [0, 1, 2, 3])
year = st.slider("Year of Purchase", 2003, 2022)
no_year = 2023 - year

fuel_type = st.selectbox("Fuel Type", ['Petrol', 'Diesel', 'CNG'])
selling_type = st.selectbox("Seller Type", ['Dealer', 'Individual'])
transmission = st.selectbox("Transmission", ['Manual', 'Automatic'])

# 🔧 One-hot encoding
fuel_map = {'Petrol': [0, 0, 1], 'Diesel': [0, 1, 0], 'CNG': [1, 0, 0]}
selling_map = {'Dealer': [1, 0], 'Individual': [0, 1]}
trans_map = {'Automatic': [1, 0], 'Manual': [0, 1]}

input_data = [
    present_price,
    driven_kms,
    owner,
    no_year,
    *fuel_map[fuel_type],
    *selling_map[selling_type],
    *trans_map[transmission]
]

# 🧮 Predict only when button is pressed
if st.button("Predict"):
    input_df = pd.DataFrame([input_data], columns=features)
    log_pred = model.predict(input_df)
    predicted_lakhs = np.expm1(log_pred)[0]  # undo log1p transformation
    price_in_inr = predicted_lakhs * 100000
    price_in_usd = price_in_inr / 83

    # 📢 Output
    st.subheader("📊 Prediction:")
    st.write(f"**≈ ${price_in_usd:,.2f} USD**")
