import streamlit as st
import pickle
import xgboost as xgb
import pandas as pd

# Load the model
with open("xgb_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Function to make predictions using the loaded model
def predict(input_data):
    # Convert input data to a pandas DataFrame (same as the format used during training)
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)
    return prediction

# Streamlit interface
st.title('Online Purchasing Prediction')

# Input fields for the user to enter data
st.sidebar.header('Parameters')

# Define the input form
def user_input_features():
    input_data = {
        'admin': st.sidebar.slider('Admin', 0, 27),
        'admin_duration': st.sidebar.slider('Admin Duration', 0, 3400),
        'info': st.sidebar.slider('Info', 0, 24),
        'info_duration': st.sidebar.slider('Info Duration', 0, 2600),
        'prod_related': st.sidebar.slider('Product Related', 0, 705),
        'prod_related_duration': st.sidebar.slider('Product Related Duration', 0, 70000),
        'bounce_rate': st.sidebar.slider('Bounce Rate', 0.0, 0.2, 0.01),
        'exit_rate': st.sidebar.slider('Exit Rate', 0.0, 0.2, 0.01),
        'page_value': st.sidebar.slider('Page Value', 0, 362),
        'special_day': st.sidebar.selectbox('Special Day', [0, 1]),
        'weekend': st.sidebar.selectbox('Weekend', [0, 1]),
        'month_Aug': st.sidebar.selectbox('Month: August', [0, 1]),
        'month_Dec': st.sidebar.selectbox('Month: December', [0, 1]),
        'month_Feb': st.sidebar.selectbox('Month: February', [0, 1]),
        'month_Jul': st.sidebar.selectbox('Month: July', [0, 1]),
        'month_June': st.sidebar.selectbox('Month: June', [0, 1]),
        'month_Mar': st.sidebar.selectbox('Month: March', [0, 1]),
        'month_May': st.sidebar.selectbox('Month: May', [0, 1]),
        'month_Nov': st.sidebar.selectbox('Month: November', [0, 1]),
        'month_Oct': st.sidebar.selectbox('Month: October', [0, 1]),
        'month_Sep': st.sidebar.selectbox('Month: September', [0, 1]),
        'visitor_type_New_Visitor': st.sidebar.selectbox('New Visitor', [0, 1]),
        'visitor_type_Other': st.sidebar.selectbox('Other Visitor', [0, 1]),
        'visitor_type_Returning_Visitor': st.sidebar.selectbox('Returning Visitor', [0, 1]),
        'os_1': st.sidebar.selectbox('OS 1', [0, 1]),
        'os_2': st.sidebar.selectbox('OS 2', [0, 1]),
        'os_3': st.sidebar.selectbox('OS 3', [0, 1]),
        'os_4': st.sidebar.selectbox('OS 4', [0, 1]),
        'os_5': st.sidebar.selectbox('OS 5', [0, 1]),
        'os_6': st.sidebar.selectbox('OS 6', [0, 1]),
        'os_7': st.sidebar.selectbox('OS 7', [0, 1]),
        'os_8': st.sidebar.selectbox('OS 8', [0, 1]),
        'browser_1': st.sidebar.selectbox('Browser 1', [0, 1]),
        'browser_2': st.sidebar.selectbox('Browser 2', [0, 1]),
        'browser_3': st.sidebar.selectbox('Browser 3', [0, 1]),
        'browser_4': st.sidebar.selectbox('Browser 4', [0, 1]),
        'browser_5': st.sidebar.selectbox('Browser 5', [0, 1]),
        'browser_6': st.sidebar.selectbox('Browser 6', [0, 1]),
        'browser_7': st.sidebar.selectbox('Browser 7', [0, 1]),
        'browser_8': st.sidebar.selectbox('Browser 8', [0, 1]),
        'browser_9': st.sidebar.selectbox('Browser 9', [0, 1]),
        'browser_10': st.sidebar.selectbox('Browser 10', [0, 1]),
        'browser_11': st.sidebar.selectbox('Browser 11', [0, 1]),
        'browser_12': st.sidebar.selectbox('Browser 12', [0, 1]),
        'browser_13': st.sidebar.selectbox('Browser 13', [0, 1]),
        'region_1': st.sidebar.selectbox('Region 1', [0, 1]),
        'region_2': st.sidebar.selectbox('Region 2', [0, 1]),
        'region_3': st.sidebar.selectbox('Region 3', [0, 1]),
        'region_4': st.sidebar.selectbox('Region 4', [0, 1]),
        'region_5': st.sidebar.selectbox('Region 5', [0, 1]),
        'region_6': st.sidebar.selectbox('Region 6', [0, 1]),
        'region_7': st.sidebar.selectbox('Region 7', [0, 1]),
        'region_8': st.sidebar.selectbox('Region 8', [0, 1]),
        'region_9': st.sidebar.selectbox('Region 9', [0, 1]),
        'traffic_type_1': st.sidebar.selectbox('Traffic Type 1', [0, 1]),
        'traffic_type_2': st.sidebar.selectbox('Traffic Type 2', [0, 1]),
        'traffic_type_3': st.sidebar.selectbox('Traffic Type 3', [0, 1]),
        'traffic_type_4': st.sidebar.selectbox('Traffic Type 4', [0, 1]),
        'traffic_type_5': st.sidebar.selectbox('Traffic Type 5', [0, 1]),
        'traffic_type_6': st.sidebar.selectbox('Traffic Type 6', [0, 1]),
        'traffic_type_7': st.sidebar.selectbox('Traffic Type 7', [0, 1]),
        'traffic_type_8': st.sidebar.selectbox('Traffic Type 8', [0, 1]),
        'traffic_type_9': st.sidebar.selectbox('Traffic Type 9', [0, 1]),
        'traffic_type_10': st.sidebar.selectbox('Traffic Type 10', [0, 1]),
        'traffic_type_11': st.sidebar.selectbox('Traffic Type 11', [0, 1]),
        'traffic_type_12': st.sidebar.selectbox('Traffic Type 12', [0, 1]),
        'traffic_type_13': st.sidebar.selectbox('Traffic Type 13', [0, 1]),
        'traffic_type_14': st.sidebar.selectbox('Traffic Type 14', [0, 1]),
        'traffic_type_15': st.sidebar.selectbox('Traffic Type 15', [0, 1]),
        'traffic_type_16': st.sidebar.selectbox('Traffic Type 16', [0, 1]),
        'traffic_type_17': st.sidebar.selectbox('Traffic Type 17', [0, 1]),
        'traffic_type_18': st.sidebar.selectbox('Traffic Type 18', [0, 1]),
        'traffic_type_19': st.sidebar.selectbox('Traffic Type 19', [0, 1]),
        'traffic_type_20': st.sidebar.selectbox('Traffic Type 20', [0, 1]),
    }
    return input_data

# Collect user input
user_input = user_input_features()

# Prediction
prediction = predict(user_input)

# Display prediction
st.subheader('Prediction')
st.write(f'Predicted Class: {prediction[0]}')  # The output class
