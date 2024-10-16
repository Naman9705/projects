import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# Function to load and preprocess the data
def load_data():
    # Load dataset from the CSV file
    data = pd.read_csv("house_data.csv")
    
    # Handle missing values by dropping rows with NaN values
    data = data.dropna()
    
    # Label Encoding for binary columns (e.g. 'yes'/'no')
    label_columns = ['airconditioning', 'basement', 'guestroom', 'mainroad', 'prefarea', 'hotwaterheating']  # Add other columns if needed
    label_encoder = LabelEncoder()
    for col in label_columns:
        data[col] = label_encoder.fit_transform(data[col])

    # One-Hot Encoding for multi-class columns (if any)
    data = pd.get_dummies(data, drop_first=True)
    
    return data

# Function to train the model
def train_model(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Function to evaluate the model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return mse, y_pred

# Streamlit app
def main():
    # Title and description
    st.title("House Price Prediction")
    st.write("This app uses a Random Forest model to predict house prices based on various features.")
    
    # Load and preprocess data
    data = load_data()
    
    # Prepare input features and target variable
    X = data.drop(columns=['price'])
    y = data['price']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    model = train_model(X_train, y_train)
    
    # Evaluate the model
    mse, y_pred = evaluate_model(model, X_test, y_test)
    
    # Display the results
    st.write(f"Mean Squared Error: {mse}")
    
    # Create input fields for prediction
    st.subheader("Enter the values for prediction:")
    
    input_data = {}
    for col in X.columns:
        input_data[col] = st.number_input(f"Enter {col}", min_value=0, step=1)
    
    # Predict on user input
    if st.button('Predict'):
        input_df = pd.DataFrame([input_data])
        
        # Apply the same preprocessing steps to the input data
        for col in label_columns:
            if col in input_df.columns:
                input_df[col] = label_encoder.transform(input_df[col])

        input_df = pd.get_dummies(input_df, drop_first=True)
        
        # Ensure input data has the same columns as training data
        input_df = input_df.reindex(columns=X.columns, fill_value=0)
        
        # Make prediction
        predicted_price = model.predict(input_df)[0]
        st.write(f"Predicted Price: {predicted_price}")
    
    # Feature importance plot
    st.subheader("Feature Importance")
    feature_importances = model.feature_importances_
    feature_names = X.columns
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)
    
    fig, ax = plt.subplots()
    sns.barplot(x='Importance', y='Feature', data=importance_df, ax=ax)
    st.pyplot(fig)

# Run the app
if __name__ == "__main__":
    main()
