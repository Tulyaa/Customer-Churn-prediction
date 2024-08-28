import streamlit as st
import pandas as pd
import pickle

# Load your pre-trained model (update the path as necessary)
with open('best_logistic_regression_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Define the feature columns based on your model
feature_columns = [
    'tenure', 'MonthlyCharges', 'TotalCharges', 'gender_Male',
    'SeniorCitizen_Yes', 'Partner_Yes', 'Dependents_Yes',
    'PhoneService_Yes', 'MultipleLines_No phone service', 'MultipleLines_Yes',
    'InternetService_Fiber optic', 'InternetService_No',
    'OnlineSecurity_No internet service', 'OnlineSecurity_Yes',
    'OnlineBackup_No internet service', 'OnlineBackup_Yes',
    'DeviceProtection_No internet service', 'DeviceProtection_Yes',
    'TechSupport_No internet service', 'TechSupport_Yes',
    'StreamingTV_No internet service', 'StreamingTV_Yes',
    'StreamingMovies_No internet service', 'StreamingMovies_Yes',
    'Contract_One year', 'Contract_Two year',
    'PaperlessBilling_Yes',
    'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check',
    'CLV', 'AvgMonthlyCharges'
]

# Function to make predictions
def predict_churn(features):
    # Create a DataFrame from the features dictionary with all required columns
    input_df = pd.DataFrame([{col: features.get(col, 0) for col in feature_columns}])
    # Predict churn
    prediction = model.predict(input_df)
    return prediction[0]

# Streamlit app
def main():
    st.title('Customer Churn Prediction')

    # User inputs
    st.header('Enter Customer Information')
    
    # Create a form for user input
    with st.form(key='input_form'):
        tenure = st.number_input('Tenure (Months)', min_value=0.0, max_value=100.0, value=12.0, step=1.0)
        MonthlyCharges = st.number_input('Monthly Charges ($)', min_value=0.0, max_value=200.0, value=60.0, step=1.0)
        TotalCharges = st.number_input('Total Charges ($)', min_value=0.0, max_value=10000.0, value=2000.0, step=10.0)
        
        gender_Male = st.selectbox('Gender', ['Female', 'Male'])
        SeniorCitizen_Yes = st.radio('Senior Citizen', ['No', 'Yes'])
        Partner_Yes = st.radio('Partner', ['No', 'Yes'])
        Dependents_Yes = st.radio('Dependents', ['No', 'Yes'])
        PhoneService_Yes = st.radio('Phone Service', ['No', 'Yes'])
        
        MultipleLines = st.selectbox('Multiple Lines', ['No phone service', 'No', 'Yes'])
        InternetService = st.selectbox('Internet Service', ['No', 'Fiber optic', 'DSL'])
        OnlineSecurity = st.selectbox('Online Security', ['No internet service', 'No', 'Yes'])
        OnlineBackup = st.selectbox('Online Backup', ['No internet service', 'No', 'Yes'])
        DeviceProtection = st.selectbox('Device Protection', ['No internet service', 'No', 'Yes'])
        TechSupport = st.selectbox('Tech Support', ['No internet service', 'No', 'Yes'])
        StreamingTV = st.selectbox('Streaming TV', ['No internet service', 'No', 'Yes'])
        StreamingMovies = st.selectbox('Streaming Movies', ['No internet service', 'No', 'Yes'])
        
        Contract = st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
        PaperlessBilling_Yes = st.radio('Paperless Billing', ['No', 'Yes'])
        
        PaymentMethod = st.selectbox('Payment Method', ['Electronic check', 'Mailed check', 'Credit card (automatic)', 'Bank transfer (automatic)'])
        CLV = st.number_input('Customer Lifetime Value ($)', min_value=0.0, max_value=10000.0, value=2000.0, step=10.0)
        AvgMonthlyCharges = st.number_input('Average Monthly Charges ($)', min_value=0.0, max_value=200.0, value=60.0, step=1.0)

        # Submit button
        submit_button = st.form_submit_button(label='Predict')

        if submit_button:
            features = {
                'tenure': tenure,
                'MonthlyCharges': MonthlyCharges,
                'TotalCharges': TotalCharges,
                'gender_Male': 1 if gender_Male == 'Male' else 0,
                'SeniorCitizen_Yes': 1 if SeniorCitizen_Yes == 'Yes' else 0,
                'Partner_Yes': 1 if Partner_Yes == 'Yes' else 0,
                'Dependents_Yes': 1 if Dependents_Yes == 'Yes' else 0,
                'PhoneService_Yes': 1 if PhoneService_Yes == 'Yes' else 0,
                'MultipleLines_No phone service': 1 if MultipleLines == 'No phone service' else 0,
                'MultipleLines_No': 1 if MultipleLines == 'No' else 0,
                'MultipleLines_Yes': 1 if MultipleLines == 'Yes' else 0,
                'InternetService_Fiber optic': 1 if InternetService == 'Fiber optic' else 0,
                'InternetService_No': 1 if InternetService == 'No' else 0,
                'OnlineSecurity_No internet service': 1 if OnlineSecurity == 'No internet service' else 0,
                'OnlineSecurity_No': 1 if OnlineSecurity == 'No' else 0,
                'OnlineSecurity_Yes': 1 if OnlineSecurity == 'Yes' else 0,
                'OnlineBackup_No internet service': 1 if OnlineBackup == 'No internet service' else 0,
                'OnlineBackup_No': 1 if OnlineBackup == 'No' else 0,
                'OnlineBackup_Yes': 1 if OnlineBackup == 'Yes' else 0,
                'DeviceProtection_No internet service': 1 if DeviceProtection == 'No internet service' else 0,
                'DeviceProtection_No': 1 if DeviceProtection == 'No' else 0,
                'DeviceProtection_Yes': 1 if DeviceProtection == 'Yes' else 0,
                'TechSupport_No internet service': 1 if TechSupport == 'No internet service' else 0,
                'TechSupport_No': 1 if TechSupport == 'No' else 0,
                'TechSupport_Yes': 1 if TechSupport == 'Yes' else 0,
                'StreamingTV_No internet service': 1 if StreamingTV == 'No internet service' else 0,
                'StreamingTV_No': 1 if StreamingTV == 'No' else 0,
                'StreamingTV_Yes': 1 if StreamingTV == 'Yes' else 0,
                'StreamingMovies_No internet service': 1 if StreamingMovies == 'No internet service' else 0,
                'StreamingMovies_No': 1 if StreamingMovies == 'No' else 0,
                'StreamingMovies_Yes': 1 if StreamingMovies == 'Yes' else 0,
                'Contract_One year': 1 if Contract == 'One year' else 0,
                'Contract_Two year': 1 if Contract == 'Two year' else 0,
                'PaperlessBilling_Yes': 1 if PaperlessBilling_Yes == 'Yes' else 0,
                'PaymentMethod_Credit card (automatic)': 1 if PaymentMethod == 'Credit card (automatic)' else 0,
                'PaymentMethod_Electronic check': 1 if PaymentMethod == 'Electronic check' else 0,
                'PaymentMethod_Mailed check': 1 if PaymentMethod == 'Mailed check' else 0,
                'CLV': CLV,
                'AvgMonthlyCharges': AvgMonthlyCharges
            }

            prediction = predict_churn(features)
            if prediction == 1:
                st.write('### Prediction')
                st.write('Based on the information provided, it is likely that the customer will churn. Consider taking actions to retain this customer.')
            else:
                st.write('### Prediction')
                st.write('Based on the information provided, it is unlikely that the customer will churn. The customer seems to be satisfied.')


if __name__ == "__main__":
    main()
