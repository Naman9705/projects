import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load the data from CSV file (assuming it's in the same repo)
@st.cache
def load_data():
    # Make sure to put the correct path to the CSV file in the repo
    data = pd.read_csv("Housing.csv")
    return data

# Apply Log Transformation to the target variable (Price)
def log_transform_target(df, target_column='price'):
    df[target_column] = np.log(df[target_column])
    return df

# Train the Random Forest Regressor Model
def train_model(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Display Feature Importance
def plot_feature_importance(model, X):
    importance = model.feature_importances_
    feature_names = X.columns
    feature_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
    feature_df = feature_df.sort_values(by='Importance', ascending=False)

    st.write("Feature Importance:")
    st.write(feature_df)

    # Plotting feature importance
    plt.figure(figsize=(10, 6))
    plt.barh(feature_df['Feature'], feature_df['Importance'])
    plt.xlabel('Importance')
    plt.title('Feature Importance')
    st.pyplot(plt)

# Main App
def main():
    st.title("House Price Prediction")
    
    # Load data
    data = load_data()

    # Apply Log Transformation on target variable (price)
    data = log_transform_target(data)
    
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
