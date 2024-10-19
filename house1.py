import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Streamlit app title
st.title("House Price Prediction App")

# Load the constant CSV file
data = pd.read_csv('Housing.csv')  # Ensure the file is in the same directory

# Preprocess categorical data (convert yes/no to 1/0)
data['mainroad'] = data['mainroad'].map({'yes': 1, 'no': 0})
data['guestroom'] = data['guestroom'].map({'yes': 1, 'no': 0})
data['basement'] = data['basement'].map({'yes': 1, 'no': 0})
data['hotwaterheating'] = data['hotwaterheating'].map({'yes': 1, 'no': 0})
data['airconditioning'] = data['airconditioning'].map({'yes': 1, 'no': 0})
data['prefarea'] = data['prefarea'].map({'yes': 1, 'no': 0})
data['furnishingstatus'] = data['furnishingstatus'].map({'unfurnished': 0, 'semi-furnished': 1, 'furnished': 2})

# Features and target
X = data.drop('price', axis=1)
y = data['price']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

# Sidebar for user inputs
st.sidebar.header("Input Features")
area = st.sidebar.number_input("Area (sqft)", value=5000)
bedrooms = st.sidebar.slider("Bedrooms", 1, 10, 3)
bathrooms = st.sidebar.slider("Bathrooms", 1, 5, 2)
stories = st.sidebar.slider("Stories", 1, 4, 2)

# Display Yes/No and map to 1/0 for model input
mainroad = st.sidebar.selectbox("Main Road", ["Yes", "No"])
guestroom = st.sidebar.selectbox("Guestroom", ["Yes", "No"])
basement = st.sidebar.selectbox("Basement", ["Yes", "No"])
hotwaterheating = st.sidebar.selectbox("Hot Water Heating", ["Yes", "No"])
airconditioning = st.sidebar.selectbox("Air Conditioning", ["Yes", "No"])
prefarea = st.sidebar.selectbox("Preferred Area", ["Yes", "No"])
parking = st.sidebar.slider("Parking", 0, 5, 2)
furnishingstatus = st.sidebar.selectbox("Furnishing Status", ["Unfurnished", "Semi-Furnished", "Furnished"])

# Map Yes/No to 1/0
mainroad = 1 if mainroad == "Yes" else 0
guestroom = 1 if guestroom == "Yes" else 0
basement = 1 if basement == "Yes" else 0
hotwaterheating = 1 if hotwaterheating == "Yes" else 0
airconditioning = 1 if airconditioning == "Yes" else 0
prefarea = 1 if prefarea == "Yes" else 0

# Map furnishing status
furnishing_map = {"Unfurnished": 0, "Semi-Furnished": 1, "Furnished": 2}
furnishingstatus = furnishing_map[furnishingstatus]

# Predict button
if st.sidebar.button("Predict Price"):
    # Create input data
    input_data = pd.DataFrame([[area, bedrooms, bathrooms, stories, mainroad, guestroom, basement,
                                hotwaterheating, airconditioning, parking, prefarea, furnishingstatus]],
                              columns=X.columns)
    
    # Make prediction
    predicted_price = model.predict(input_data)[0]
    
    # Display the result
    st.write(f"Predicted House Price: ₹{predicted_price:,.2f}")

# Show model evaluation metrics
st.subheader("Model Performance")
