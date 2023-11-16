# 03372025__Churning_Customers
AI Model To Predict Customer Churn

# Churn Prediction Model

This repository contains code for a churn prediction model developed for a telecom company. The model predicts customer churn, allowing proactive strategies to retain customers.

## Dependencies

The code is written in Python and requires the following libraries:
- TensorFlow
- Keras
- Pandas
- Scikit-learn
- Joblib
- Matplotlib
- Flask

## Setup Instructions

1. Install the required dependencies using pip:
    ```

2. Execute the provided Jupyter Notebook `Malcolm_Clottey_Assignment_3.ipynb`. This notebook contains the entire data preprocessing, model training, and evaluation pipeline.

## Files and Usage

- `Malcolm_Clottey_Assignment_3.ipynb`: Contains the complete code for data preprocessing, model training (Random Forest Classifier, MLP Classifier), hyperparameter tuning using GridSearchCV, and model evaluation.
- `one_hot_encoder.pkl`: Serialized OneHotEncoder object used for categorical encoding.
- `scaler.pkl`: Serialized StandardScaler object used for feature scaling.
- `model.h5`: Saved Keras model for neural network-based churn prediction.
- `app.py`: Flask application for deploying the churn prediction model as a web service.
- `index.html`: HTML file for the web interface of the churn prediction service.
- `static/main.css`: CSS file for styling the web interface.

## Usage Instructions

1. Run the Flask application:
    ```
    python app.py
    ```

2. Open a web browser and navigate to `http://127.0.0.1:5000/` to access the churn prediction web interface.

3. Enter customer details and click 'Predict' to get churn predictions.

## Developer

- Developed by Malottey (Malcolm Clottey)


