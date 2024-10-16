import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset (You can change the path if the CSV is elsewhere)
@st.cache
def load_data():
    data = pd.read_csv('Housing.csv')
    return data

data = load_data()

# Sidebar: Show available columns and their types
st.sidebar.title("Data Overview")
st.sidebar.write("Dataset Columns and Types:")
st.sidebar.write(data.dtypes)

# Display dataset information and basic statistics
st.write("### House Price Prediction Dataset")
st.write(data.head())
st.write("### Summary Statistics")
st.write(data.describe())

# Handle categorical features: Convert 0, 1 features to Yes/No
binary_columns = ['guestroom', 'mainroad', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea', 'furnishingstatus']

for col in binary_columns:
    data[col] = data[col].map({0: 'No', 1: 'Yes'})

# Display feature engineering options
st.sidebar.title("Feature Engineering")
st.sidebar.write("Categorical features like 'guestroom' are now shown as 'Yes' or 'No'.")

# Feature Selection: Drop target column and any unnecessary columns
X = data.drop(columns=['price'])
y = data['price']

# Apply log transformation to the target variable (house price)
y_log = np.log1p(y)  # log1p is log(1 + y), useful when the target has zeros

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y_log, test_size=0.2, random_state=42)

# Train Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions and Model Evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Show predictions and performance metrics
st.write("### Model Performance")
st.write(f"Mean Squared Error (MSE): {mse:.2f}")
st.write(f"R² Score: {r2:.2f}")

# Display feature importance for Random Forest model
st.write("### Feature Importance")
feature_importances = model.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importances
})
st.write(feature_importance_df.sort_values(by='Importance', ascending=False))

# Predict a new house price with user inputs (optional)
st.sidebar.title("House Price Prediction")

# User input: Allow user to input values for the features
guestroom = st.sidebar.selectbox("Guestroom (Yes/No)", ["Yes", "No"])
mainroad = st.sidebar.selectbox("Mainroad (Yes/No)", ["Yes", "No"])
basement = st.sidebar.selectbox("Basement (Yes/No)", ["Yes", "No"])
hotwaterheating = st.sidebar.selectbox("Hotwaterheating (Yes/No)", ["Yes", "No"])
airconditioning = st.sidebar.selectbox("Airconditioning (Yes/No)", ["Yes", "No"])
prefarea = st.sidebar.selectbox("Prefarea (Yes/No)", ["Yes", "No"])
furnishingstatus = st.sidebar.selectbox("Furnishing Status (Yes/No)", ["Yes", "No"])

# Map user inputs to numeric values (0 for No, 1 for Yes)
input_data = {
    'guestroom': 1 if guestroom == "Yes" else 0,
    'mainroad': 1 if mainroad == "Yes" else 0,
    'basement': 1 if basement == "Yes" else 0,
    'hotwaterheating': 1 if hotwaterheating == "Yes" else 0,
    'airconditioning': 1 if airconditioning == "Yes" else 0,
    'prefarea': 1 if prefarea == "Yes" else 0,
    'furnishingstatus': 1 if furnishingstatus == "Yes" else 0,
    # You can add other numerical features (e.g., area, bedrooms, etc.) here
}

# Convert input data into a DataFrame for prediction
input_df = pd.DataFrame([input_data])

# Predict the house price using the trained model
predicted_price_log = model.predict(input_df)[0]
predicted_price = np.expm1(predicted_price_log)  # Convert log-transformed value back to original scale

st.write(f"### Predicted House Price: ₹{predicted_price:,.2f}")

# Show a correlation heatmap
st.write("### Correlation Heatmap")
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
st.pyplot()

