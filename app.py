import streamlit as st
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression

# Streamlit application
st.title('Loan Eligibility Prediction')

# Input fields for user
gender = st.selectbox('Gender', ['Male', 'Female'])
married = st.selectbox('Married', ['Yes', 'No'])
dependents = st.selectbox('Dependents', ['0', '1', '2', '3+'])
education = st.selectbox('Education', ['Graduate', 'Not Graduate'])
self_employed = st.selectbox('Self Employed', ['Yes', 'No'])
applicant_income = st.number_input('Applicant Income', min_value=0)
coapplicant_income = st.number_input('Coapplicant Income', min_value=0)
loan_amount = st.number_input('Loan Amount', min_value=0)
loan_amount_term = st.number_input('Loan Amount Term', min_value=0)
credit_history = st.selectbox('Credit History', [1.0, 0.0])
property_area = st.selectbox('Property Area', ['Urban', 'Semiurban', 'Rural'])

# Collect user input into a DataFrame
user_input = pd.DataFrame({
    'Gender': [1 if gender == 'Male' else 0],
    'Married': [1 if married == 'Yes' else 0],
    'Dependents': [3 if dependents == '3+' else int(dependents)],
    'Education': [1 if education == 'Graduate' else 0],
    'Self_Employed': [1 if self_employed == 'Yes' else 0],
    'ApplicantIncome': [applicant_income],
    'CoapplicantIncome': [coapplicant_income],
    'LoanAmount': [loan_amount],
    'Loan_Amount_Term': [loan_amount_term],
    'Credit_History': [credit_history],
    'Property_Area': [2 if property_area == 'Urban' else 1 if property_area == 'Semiurban' else 0]
})

# Load the trained model

with open('trained_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Make prediction
if st.button('Predict'):
    prediction = model.predict(user_input)
    st.write(f'Loan Eligibility: {"Eligible" if prediction[0] == 1 else "Not Eligible"}')
