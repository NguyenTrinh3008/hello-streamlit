import numpy as np
import pickle
import streamlit as st


# loading the saved model
loaded_model = pickle.load(open('E:\AI4E-Project\credit_classifier_model.sav', 'rb'))


# creating a function for Prediction

def credit_prediction(input_data):

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data = input_data_as_numpy_array.reshape(1,-1)

    adaboost_predictions_new = loaded_model.predict(input_data)

    # Print the predictions
    print(adaboost_predictions_new)

    if (adaboost_predictions_new[0] == 0):
        return'The person is good loan'
    else:
        return'The person is bad loan'
  
def main():    
    # giving a title
    st.title('Credit Prediction Web App')
    
    
    # getting the input data from the user
    duration = st.text_input('Duration')
    credit_amount = st.text_input('Credit Amount')
    installment_rate = st.text_input('Installment Rate')
    residence_since = st.text_input('Residence Since')
    age = st.text_input('Age')
    number_of_existcr = st.text_input('Number of Existing Credits')
    number_of_dependents = st.text_input('Number of Dependents')
    telephon = st.text_input('Telephone')
    
    foreign = st.checkbox('Foreign Worker')

    account_bal_neg_bal = st.checkbox('Account Balance - Negative Balance')
    account_bal_no_acc = st.checkbox('Account Balance - No Account')
    account_bal_positive_bal = st.checkbox('Account Balance - Positive Balance')

    payment_status_A30 = st.checkbox('Payment Status - No credits taken/All credits paid back duly')
    payment_status_A31 = st.checkbox('Payment Status - All credits at this bank paid back duly')
    payment_status_A32 = st.checkbox('Payment Status - Existing credits paid back duly till now')
    payment_status_A33 = st.checkbox('Payment Status - Delay in paying off in the past')
    payment_status_A34 = st.checkbox('Payment Status - Critical account/Other credits existing (not at this bank)')

    purpose_A40 = st.checkbox('Purpose - car (new)')
    purpose_A41 = st.checkbox('Purpose - car (used)')
    purpose_A410 = st.checkbox('Purpose - others')
    purpose_A42 = st.checkbox('Purpose - furniture/equipment')
    purpose_A43 = st.checkbox('Purpose - radio/television')
    purpose_A44 = st.checkbox('Purpose - domestic appliances')
    purpose_A45 = st.checkbox('Purpose - repairs')
    purpose_A46 = st.checkbox('Purpose - education')
    purpose_A48 = st.checkbox('Purpose - retraining')
    purpose_A49 = st.checkbox('Purpose - business')

    savings_bond_value_A61 = st.checkbox('Savings Bond Value - ... <  100 DM')
    savings_bond_value_A62 = st.checkbox('Savings Bond Value - 100 <= ... <  500 DM')
    savings_bond_value_A63 = st.checkbox('Savings Bond Value -  500 <= ... < 1000 DM')
    savings_bond_value_A64 = st.checkbox('Savings Bond Value - .. >= 1000 DM')
    savings_bond_value_A65 = st.checkbox('Savings Bond Value - unknown/ no savings account')

    employed_since_A71 = st.checkbox('Employed Since - unemployed')
    employed_since_A72 = st.checkbox('Employed Since - ... < 1 year')
    employed_since_A73 = st.checkbox('Employed Since - 1  <= ... < 4 years')
    employed_since_A74 = st.checkbox('Employed Since - 4  <= ... < 7 years')
    employed_since_A75 = st.checkbox('Employed Since - .. >= 7 years')

    sex_marital_A91 = st.checkbox('Sex/Marital Status -  male: divorced/separated')
    sex_marital_A92 = st.checkbox('Sex/Marital Status - female: divorced/separated/married')
    sex_marital_A93 = st.checkbox('Sex/Marital Status - male: single')
    sex_marital_A94 = st.checkbox('Sex/Marital Status - male: married/widowed')

    guarantor_A101 = st.checkbox('Guarantor - none')
    guarantor_A102 = st.checkbox('Guarantor - co-applicant')
    guarantor_A103 = st.checkbox('Guarantor - guarantor')

    most_valuable_asset_car = st.checkbox('Most Valuable Asset - Car')
    most_valuable_asset_life_insurance = st.checkbox('Most Valuable Asset - Life Insurance')
    most_valuable_asset_none = st.checkbox('Most Valuable Asset - None')
    most_valuable_asset_real_estate = st.checkbox('Most Valuable Asset - Real Estate')

    concurrent_credits_A141 = st.checkbox('Concurrent Credits - bank')
    concurrent_credits_A142 = st.checkbox('Concurrent Credits - stores')
    concurrent_credits_A143 = st.checkbox('Concurrent Credits - none')

    type_of_housing_A151 = st.checkbox('Type of Housing - rent')
    type_of_housing_A152 = st.checkbox('Type of Housing - own')
    type_of_housing_A153 = st.checkbox('Type of Housing - for free')

    job_highly_skilled = st.checkbox('Job - Highly Skilled')
    job_skilled = st.checkbox('Job - Skilled')
    job_unskilled = st.checkbox('Job - Unskilled')

    
    
    # code for Prediction
    diagnosis = ''
    
    
    # creating a button for Predictio

    # Button for prediction
    if st.button('Credit Test Result'):
        try:
        # Convert inputs to appropriate numeric types
            input_features = [
                float(duration), float(credit_amount), float(installment_rate), float(residence_since), float(age),
                int(number_of_existcr), int(number_of_dependents), int(telephon), int(foreign),
                int(account_bal_neg_bal), int(account_bal_no_acc), int(account_bal_positive_bal),
                int(payment_status_A30), int(payment_status_A31), int(payment_status_A32), int(payment_status_A33), int(payment_status_A34),
                int(purpose_A40), int(purpose_A41), int(purpose_A410), int(purpose_A42), int(purpose_A43), int(purpose_A44),
                int(purpose_A45), int(purpose_A46), int(purpose_A48), int(purpose_A49),
                int(savings_bond_value_A61), int(savings_bond_value_A62), int(savings_bond_value_A63), int(savings_bond_value_A64), int(savings_bond_value_A65),
                int(employed_since_A71), int(employed_since_A72), int(employed_since_A73), int(employed_since_A74), int(employed_since_A75),
                int(sex_marital_A91), int(sex_marital_A92), int(sex_marital_A93), int(sex_marital_A94),
                int(guarantor_A101), int(guarantor_A102), int(guarantor_A103),
                int(most_valuable_asset_car), int(most_valuable_asset_life_insurance), int(most_valuable_asset_none), int(most_valuable_asset_real_estate),
                int(concurrent_credits_A141), int(concurrent_credits_A142), int(concurrent_credits_A143),
                int(type_of_housing_A151), int(type_of_housing_A152), int(type_of_housing_A153),
                int(job_highly_skilled), int(job_skilled), int(job_unskilled)
            ]
        
            # Prediction function
            diagnosis = credit_prediction(input_features)
            st.success(diagnosis)
        except ValueError as e:
            st.error(f"Error in input conversion: {e}")


if __name__ == '__main__':
    main()