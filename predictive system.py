import joblib
import numpy as np
import pickle
# Load the model
loaded_model = pickle.load(open('E:\AI4E-Project\credit_classifier_model.sav', 'rb'))
def credit_prediction(input_data):
    input_data = np.random.rand(57)
    input_data = input_data.reshape(1, -1)
    # Make predictions on the new dataset
    adaboost_predictions_new = loaded_model.predict(input_data)

    # Print the predictions
    print(adaboost_predictions_new)

    if (adaboost_predictions_new[0] == 0):
        print('The person is good loan')
    else:
        print('The person is bad loan')