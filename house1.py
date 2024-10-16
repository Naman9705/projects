import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load the dataset (constant file path)
@st.cache
def load_data():
    data = pd.read_csv('Housing.csv')
    return data

# Load the dataset
data = load_data()

# Prepare the data
binary_columns = ['guestroom', 'mainroad', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea', 'furnishingstatus']

# Convert binary columns to Yes/No
for col in binary_columns:
    data[col] = data[col].map({0: 'No', 1: 'Yes'})

# Feature Selection: Drop target column and any unnecessary columns
X = data.drop(columns=['price'])
y = data['price']

# Apply log transformation to the target variable (house price)
y_log = np.log1p(y)  # log1p is log(1 + y), useful when the target has zeros

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y_log, test_size=0.2, random_state=42)

# Train Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=15, min_samples_split=10)
model.fit(X_train, y_train)

# Predictions and Model Evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

# Streamlit Sidebar for Inputs
st.sidebar.title("House Price Prediction Inputs")

area = st.sidebar.number_input("Area (in square feet)", min_value=1, step=1)
bedrooms = st.sidebar.number_input("Bedrooms", min_value=1, step=1)
bathrooms = st.sidebar.number_input("Bathrooms", min_value=1, step=1)
stories = st.sidebar.number_input("Stories", min_value=1, step=1)

guestroom = st.sidebar.selectbox("Guestroom (Yes/No)", ["Yes", "No"])
mainroad = st.sidebar.selectbox("Mainroad (Yes/No)", ["Yes", "No"])
basement = st.sidebar.selectbox("Basement (Yes/No)", ["Yes", "No"])
hotwaterheating = st.sidebar.selectbox("Hotwaterheating (Yes/No)", ["Yes", "No"])
airconditioning = st.sidebar.selectbox("Airconditioning (Yes/No)", ["Yes", "No"])
prefarea = st.sidebar.selectbox("Prefarea (Yes/No)", ["Yes", "No"])
furnishingstatus = st.sidebar.selectbox("Furnishing Status (Yes/No)", ["Yes", "No"])

# Map user inputs to numeric values (0 for No, 1 for Yes)
input_data = {
    'area': area,
    'bedrooms': bedrooms,
    'bathrooms': bathrooms,
    'stories': stories,
    'guestroom': 1 if guestroom == "Yes" else 0,
    'mainroad': 1 if mainroad == "Yes" else 0,
    'basement': 1 if basement == "Yes" else 0,
    'hotwaterheating': 1 if hotwaterheating == "Yes" else 0,
    'airconditioning': 1 if airconditioning == "Yes" else 0,
    'prefarea': 1 if prefarea == "Yes" else 0,
    'furnishingstatus': 1 if furnishingstatus == "Yes" else 0,
}

# Convert input data into a DataFrame for prediction
input_df = pd.DataFrame([input_data])

# Align the columns of input_df with the training data (X)
input_df = input_df[X.columns]

# Predict the house price using the trained model
predicted_price_log = model.predict(input_df)[0]
predicted_price = np.expm1(predicted_price_log)  # Convert log-transformed value back to original scale

# Display the results in the main window
st.write(f"### Predicted House Price: ₹{predicted_price:,.2f}")
st.write(f"### Mean Squared Error (MSE): {mse:.2f}")
