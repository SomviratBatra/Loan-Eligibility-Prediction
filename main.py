# # import streamlit as st
# # import pandas as pd
# # import os
# # import matplotlib.pyplot as plt
# # import seaborn as sns
# #
# # # Set the page config
# # st.set_page_config(page_title='Data Visualizer',
# #                    layout='centered',
# #                    page_icon='📊')
# #
# # # Title
# # st.title('📊  Data Visualizer')
# #
# # working_dir = os.path.dirname(os.path.abspath(__file__))
# #
# # # Specify the folder where your CSV files are located
# # folder_path = f"{working_dir}/data"  # Update this to your folder path
# #
# # # List all files in the folder
# # files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
# #
# # # Dropdown to select a file
# # selected_file = st.selectbox('Select a file', files, index=None)
# #
# # if selected_file:
# #     # Construct the full path to the file
# #     file_path = os.path.join(folder_path, selected_file)
# #
# #     # Read the selected CSV file
# #     df = pd.read_csv(file_path)
# #
# #     col1, col2 = st.columns(2)
# #
# #     columns = df.columns.tolist()
# #
# #     with col1:
# #         st.write("")
# #         st.write(df.head())
# #
# #     with col2:
# #         # Allow the user to select columns for plotting
# #         x_axis = st.selectbox('Select the X-axis', options=columns+["None"])
# #         y_axis = st.selectbox('Select the Y-axis', options=columns+["None"])
# #
# #         plot_list = ['Line Plot', 'Bar Chart', 'Scatter Plot', 'Distribution Plot', 'Count Plot']
# #         # Allow the user to select the type of plot
# #         plot_type = st.selectbox('Select the type of plot', options=plot_list)
# #
# #     # Generate the plot based on user selection
# #     if st.button('Generate Plot'):
# #
# #         fig, ax = plt.subplots(figsize=(6, 4))
# #
# #         if plot_type == 'Line Plot':
# #             sns.lineplot(x=df[x_axis], y=df[y_axis], ax=ax)
# #         elif plot_type == 'Bar Chart':
# #             sns.barplot(x=df[x_axis], y=df[y_axis], ax=ax)
# #         elif plot_type == 'Scatter Plot':
# #             sns.scatterplot(x=df[x_axis], y=df[y_axis], ax=ax)
# #         elif plot_type == 'Distribution Plot':
# #             sns.histplot(df[x_axis], kde=True, ax=ax)
# #             y_axis='Density'
# #         elif plot_type == 'Count Plot':
# #             sns.countplot(x=df[x_axis], ax=ax)
# #             y_axis = 'Count'
# #
# #         # Adjust label sizes
# #         ax.tick_params(axis='x', labelsize=10)  # Adjust x-axis label size
# #         ax.tick_params(axis='y', labelsize=10)  # Adjust y-axis label size
# #
# #         # Adjust title and axis labels with a smaller font size
# #         plt.title(f'{plot_type} of {y_axis} vs {x_axis}', fontsize=12)
# #         plt.xlabel(x_axis, fontsize=10)
# #         plt.ylabel(y_axis, fontsize=10)
# #
# #         # Show the results
# #         st.pyplot(fig)
# #
# #
# # import streamlit as st
# #
# # # Title of the app
# # st.title("Multiple Inputs Form Example")
# #
# # # Form to accept multiple inputs
# # with st.form(key='input_form'):
# #     name = st.text_input(label="Enter your name")
# #     age = st.number_input(label="Enter your age", min_value=0)
# #     email = st.text_input(label="Enter your email")
# #     address = st.text_area(label="Enter your address")
# #
# #     # Button to submit the form
# #     submit_button = st.form_submit_button(label='Submit')
# #
# # # Display the submitted values
# # if submit_button:
# #     st.write(f"Name: {name}")
# #     st.write(f"Age: {age}")
# #     st.write(f"Email: {email}")
# #     st.write(f"Address: {address}")
# #     st.success("Form submitted successfully!")
#
# import streamlit as st
# import pandas as pd
# import pickle
# from sklearn.linear_model import LogisticRegression
#
#
# # Streamlit application
# st.title('Loan Eligibility Prediction')
#
# # Input fields for user
# gender = st.selectbox('Gender', ['Male', 'Female'])
# married = st.selectbox('Married', ['Yes', 'No'])
# dependents = st.selectbox('Dependents', ['0', '1', '2', '3+'])
# education = st.selectbox('Education', ['Graduate', 'Not Graduate'])
# self_employed = st.selectbox('Self Employed', ['Yes', 'No'])
# applicant_income = st.number_input('Applicant Income', min_value=0)
# coapplicant_income = st.number_input('Coapplicant Income', min_value=0)
# loan_amount = st.number_input('Loan Amount', min_value=0)
# loan_amount_term = st.number_input('Loan Amount Term', min_value=0)
# credit_history = st.selectbox('Credit History', [1.0, 0.0])
# property_area = st.selectbox('Property Area', ['Urban', 'Semiurban', 'Rural'])
#
# # Collect user input into a DataFrame
# user_input = pd.DataFrame({
#     'Gender': [1 if gender == 'Male' else 0],
#     'Married': [1 if married == 'Yes' else 0],
#     'Dependents': [3 if dependents == '3+' else int(dependents)],
#     'Education': [1 if education == 'Graduate' else 0],
#     'Self_Employed': [1 if self_employed == 'Yes' else 0],
#     'ApplicantIncome': [applicant_income],
#     'CoapplicantIncome': [coapplicant_income],
#     'LoanAmount': [loan_amount],
#     'Loan_Amount_Term': [loan_amount_term],
#     'Credit_History': [credit_history],
#     'Property_Area': [2 if property_area == 'Urban' else 1 if property_area == 'Semiurban' else 0]
# })
#
# # Load the trained model
# with open('trained_modell.pkl','rb') as file:  # Open in binary mode
#   model = pickle.load(file)
# import joblib
#
# # Read model from file
# # try:
# #     with open('trained_model.sav', 'rb') as file:
# #         model = pickle.load(file)
# #         print("Model loaded successfully:", model)
# # except pickle.UnpicklingError:
# #     print("Error: The file is not a valid pickle file.")
# # except EOFError:
# #     print("Error: End of file reached unexpectedly.")
# # except Exception as e:
# #     print(f"An unexpected error occurred: {e}")
#
# # Make prediction
# # if st.button('Predict'):
# #     prediction = model.predict(user_input)
# #     st.write(f'Loan Eligibility: {"Eligible" if prediction[0] == 1 else "Not Eligible"}')
#
# if st.button('Predict'):
#     try:
#         # Make predictions
#         prediction = model.predict(user_input)
#         st.success(f'The prediction is {prediction}')
#     except Exception as e:
#         st.error(f'Error: {e}')
#
#
#
#
#
#
#
#
#

import streamlit as st
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression

# Streamlit application title
st.title('📊 Loan Eligibility Prediction')

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
try:
    with open('trained_model.pkl', 'rb') as file:
        model = pickle.load(file)
    if not hasattr(model, 'coef_'):
        st.error("Model is not trained. Please train the model before using it.")
except FileNotFoundError:
    st.error("Model file not found. Please make sure 'trained_model.pkl' exists.")
except Exception as e:
    st.error(f"Error loading model: {e}")

# Make prediction
if st.button('Predict'):
    try:
        if hasattr(model, 'coef_'):
            prediction = model.predict(user_input)
            st.success(f'Loan Eligibility Prediction: {"Eligible" if prediction[0] == 1 else "Not Eligible"}')
        else:
            st.error("Model is not trained. Please train the model before using it.")
    except Exception as e:
        st.error(f'Error predicting loan eligibility: {e}')