import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import OneHotEncoder

# Load the trained Random Forest model
model = joblib.load('rf_model.joblib')

# Load the original dataframe to get column names for encoding consistency
# The tourism.csv file is expected to be in the same directory as app.py
original_df = pd.read_csv('tourism.csv')

# Load the pre-fitted OneHotEncoder
encoder = joblib.load('encoder.joblib')

# Identify categorical columns from the original dataset (for feature column definition)
categorical_cols = original_df.select_dtypes(include='object').columns.tolist()

# Define the expected order of numerical features based on X_train
# This is important for consistent input to the model
# Assuming 'ProdTaken', 'Unnamed: 0', 'CustomerID' are dropped
numerical_cols = [col for col in original_df.columns if col not in categorical_cols + ['ProdTaken', 'Unnamed: 0', 'CustomerID']]

st.title('Wellness Tourism Package Purchase Prediction')
st.write('Enter customer details to predict if they will purchase the package.')

# Input fields for the Streamlit app
# Numerical inputs
age = st.number_input('Age', min_value=18, max_value=100, value=30)
duration_of_pitch = st.number_input('Duration of Pitch (minutes)', min_value=1.0, max_value=60.0, value=10.0)
number_of_person_visiting = st.number_input('Number of Persons Visiting', min_value=1, max_value=10, value=1)
number_of_followups = st.number_input('NumberOfFollowups', min_value=0.0, max_value=10.0, value=2.0)
preferred_property_star = st.number_input('Preferred Property Star (1-5)', min_value=1.0, max_value=5.0, value=3.0)
number_of_trips = st.number_input('Number of Trips Annually', min_value=0.0, max_value=50.0, value=5.0)
pitch_satisfaction_score = st.number_input('Pitch Satisfaction Score (1-5)', min_value=1, max_value=5, value=3)
own_car = st.selectbox('Own Car', options=[0, 1], format_func=lambda x: 'Yes' if x==1 else 'No')
number_of_children_visiting = st.number_input('Number of Children Visiting', min_value=0.0, max_value=5.0, value=0.0)
monthly_income = st.number_input('Monthly Income', min_value=0.0, value=50000.0)

# Categorical inputs
type_of_contact = st.selectbox('Type of Contact', options=original_df['TypeofContact'].unique())
occupation = st.selectbox('Occupation', options=original_df['Occupation'].unique())
gender = st.selectbox('Gender', options=original_df['Gender'].unique())
product_pitched = st.selectbox('Product Pitched', options=original_df['ProductPitched'].unique())
marital_status = st.selectbox('Marital Status', options=original_df['MaritalStatus'].unique())
designation = st.selectbox('Designation', options=original_df['Designation'].unique())

city_tier = st.selectbox('City Tier', options=original_df['CityTier'].unique())
passport = st.selectbox('Passport', options=original_df['Passport'].unique(), format_func=lambda x: 'Yes' if x==1 else 'No')


if st.button('Predict'):
    # Create a DataFrame from current inputs for prediction
    input_data = pd.DataFrame({
        'Age': [age],
        'DurationOfPitch': [duration_of_pitch],
        'NumberOfPersonVisiting': [number_of_person_visiting],
        'NumberOfFollowups': [number_of_followups],
        'PreferredPropertyStar': [preferred_property_star],
        'NumberOfTrips': [number_of_trips],
        'PitchSatisfactionScore': [pitch_satisfaction_score],
        'OwnCar': [own_car],
        'NumberOfChildrenVisiting': [number_of_children_visiting],
        'MonthlyIncome': [monthly_income],
        'CityTier': [city_tier],
        'Passport': [passport],
        'TypeofContact': [type_of_contact],
        'Occupation': [occupation],
        'Gender': [gender],
        'ProductPitched': [product_pitched],
        'MaritalStatus': [marital_status],
        'Designation': [designation]
    })

    # Separate numerical and categorical features from input_data
    input_categorical_data = input_data[categorical_cols]
    input_numerical_data = input_data[numerical_cols]

    # Encode categorical features using the loaded encoder
    encoded_input_features = encoder.transform(input_categorical_data)
    encoded_input_df = pd.DataFrame(encoded_input_features, columns=encoder.get_feature_names_out(categorical_cols), index=input_data.index)

    # Combine numerical and encoded categorical features
    final_input_df = pd.concat([input_numerical_data, encoded_input_df], axis=1)

    # Ensure column order matches training data (important for consistent predictions)
    # This assumes X_train.columns reflects the order after encoding and dropping
    # For a robust solution, you'd save X_train.columns and use it here
    # For now, let's assume the order is consistent due to how OneHotEncoder works

    # Make prediction
    prediction = model.predict(final_input_df)
    prediction_proba = model.predict_proba(final_input_df)[:, 1]

    if prediction[0] == 1:
        st.success(f'Prediction: Customer is LIKELY to purchase the package (Probability: {prediction_proba[0]:.2f})')
    else:
        st.warning(f'Prediction: Customer is UNLIKELY to purchase the package (Probability: {prediction_proba[0]:.2f})')
