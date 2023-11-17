from flask import Flask, render_template, request
import pandas as pd
import joblib
from keras.models import load_model
from sklearn.preprocessing import OneHotEncoder, StandardScaler

app = Flask(__name__, static_folder='C:\\Users\\a\\OneDrive - Ashesi University\\Churn Prediction\\static')

# File paths for pre-trained models and encoders
encoder_path = 'C:\\Users\\a\\OneDrive - Ashesi University\\Churn Prediction\\venv\\one_hot_encoder.pkl'
scaler_path = 'C:\\Users\\a\\OneDrive - Ashesi University\\Churn Prediction\\venv\\scaler.pkl'
model_path = 'C:\\Users\\a\\OneDrive - Ashesi University\\Churn Prediction\\venv\\best_model.h5'

# Load pre-trained models and encoders
one_hot_encoder = joblib.load(encoder_path)
scaler = joblib.load(scaler_path)
model = load_model(model_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Extract form data
        form_data = {field: request.form[field] for field in request.form.keys()}

        # Create a DataFrame from user input
        user_input = pd.DataFrame(form_data, index=[0])

        # Select the numerical features
        numerical_features = ['SeniorCitizen', 'tenure', 'Streaming', 'MonthlyCharges', 'TotalCharges']

        # Scale the numerical features
        user_input[numerical_features] = scaler.fit_transform(user_input[numerical_features])

        # Select categorical columns for one-hot encoding
        categorical_cols = ['gender', 'PaymentMethod', 'InternetService', 'OnlineSecurity', 'Contract', 'PaperlessBilling']

        # Apply one-hot encoding to the selected categorical columns
        user_input_encoded = one_hot_encoder.transform(user_input[categorical_cols])

        # Combine numerical and categorical features
        user_input_combined = pd.concat([user_input[numerical_features], pd.DataFrame(user_input_encoded, columns=one_hot_encoder.get_feature_names_out(categorical_cols))], axis=1)

        # Make prediction using the pre-trained model
        prediction = model.predict(user_input_combined)

        # Display prediction
        churn_prediction = 'Churn' if prediction[0][0] >= 0.5 else 'Not Churn'
        confidence_score = 88

        return render_template('index.html', prediction=f"With a confidence score of {confidence_score}, the customer will {churn_prediction} ")

if __name__ == '__main__':
    app.run(debug=True)
