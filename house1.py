import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Function to load and preprocess the data
def load_data():
    # Load dataset from the CSV file
    data = pd.read_csv("house_data.csv")
    
    # Handle missing values by dropping rows with NaN values
    data = data.dropna()

    # Apply Log Transformation on the target variable (price)
    data['price'] = np.log1p(data['price'])

    return data

# Function to train the model
def train_model(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Function to plot feature importance
def plot_feature_importance(model, X):
    feature_importance = model.feature_importances_
    feature_names = X.columns

    # Create a DataFrame of feature importance
    feature_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    }).sort_values(by='Importance', ascending=False)

    # Plot feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_df)
    plt.title('Feature Importance')
    plt.show()

# Main function to build the Streamlit interface
def main():
    st.title("House Price Prediction")
    
    # Load and preprocess data
    data = load_data()

    # Prepare input features and target variable
    X = data.drop(columns=['price'])
    y = data['price']

    # Split the dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = train_model(X_train, y_train)

    # Feature importance
    plot_feature_importance(model, X)

    # Sidebar Input Fields for Prediction
    st.sidebar.header("Input Features for Prediction")
    area = st.sidebar.number_input('Area (in sqft)', min_value=1, step=1)
    bedrooms = st.sidebar.number_input('Bedrooms', min_value=1, step=1)
    bathrooms = st.sidebar.number_input('Bathrooms', min_value=1, step=1)
    stories = st.sidebar.number_input('Stories', min_value=1, step=1)
    mainroad = st.sidebar.selectbox('Mainroad', ['No', 'Yes'])
    guestroom = st.sidebar.selectbox('Guestroom', ['No', 'Yes'])
    basement = st.sidebar.selectbox('Basement', ['No', 'Yes'])
    hotwaterheating = st.sidebar.selectbox('Hotwaterheating', ['No', 'Yes'])
    airconditioning = st.sidebar.selectbox('Airconditioning', ['No', 'Yes'])
    parking = st.sidebar.number_input('Parking', min_value=0, step=1)
    prefarea = st.sidebar.selectbox('Preferred Area', ['No', 'Yes'])
    furnishingstatus = st.sidebar.selectbox('Furnishing Status', ['Unfurnished', 'Semi-Furnished', 'Furnished'])

    # Map categorical inputs to 0 or 1
    mainroad = 1 if mainroad == 'Yes' else 0
    guestroom = 1 if guestroom == 'Yes' else 0
    basement = 1 if basement == 'Yes' else 0
    hotwaterheating = 1 if hotwaterheating == 'Yes' else 0
    airconditioning = 1 if airconditioning == 'Yes' else 0
    prefarea = 1 if prefarea == 'Yes' else 0

    # Prepare input data as a DataFrame
    input_data = pd.DataFrame({
        'area': [area],
        'bedrooms': [bedrooms],
        'bathrooms': [bathrooms],
        'stories': [stories],
        'mainroad': [mainroad],
        'guestroom': [guestroom],
        'basement': [basement],
        'hotwaterheating': [hotwaterheating],
        'airconditioning': [airconditioning],
        'parking': [parking],
        'prefarea': [prefarea],
        'furnishingstatus': [furnishingstatus]
    })

    # Make prediction when user presses button
    if st.sidebar.button("Predict House Price"):
        predicted_price_log = model.predict(input_data)[0]
        
        # Inverse log transformation
        predicted_price = np.exp(predicted_price_log)

        # Display predicted price
        st.subheader(f"Predicted House Price: ₹{predicted_price:,.2f}")

        # Model evaluation on test set (MSE)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        st.subheader(f"Mean Squared Error (MSE) on Test Set: {mse:,.2f}")

# Run the app
if __name__ == "__main__":
    main()
