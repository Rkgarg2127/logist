from flask import Flask, request, jsonify
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Initialize Flask app
app = Flask(__name__)

# Load the saved models and scaler
model_logistic_regression = joblib.load('logistic_regression_model.pkl')
model_random_forest = joblib.load('random_forest_model.pkl')
model_gradient_boosting = joblib.load('gradient_boosting_model.pkl')
scaler = joblib.load('scaler.pkl')

# Preprocess incoming data for prediction
def preprocess_data(data):
    # Convert categorical columns to dummy variables
    categorical_columns = ['Vehicle Type', 'Weather Conditions', 'Traffic Conditions', 'Origin', 'Destination']
    data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)
    
    # Ensure the correct columns are present (based on your model's features)
    expected_columns = model_gradient_boosting.feature_importances_.shape[0]
    if data.shape[1] != expected_columns:
        raise ValueError(f"Incorrect number of features. Expected {expected_columns} features.")
    
    # Scale the data using the pre-trained scaler
    data_scaled = scaler.transform(data)
    return data_scaled

# Define prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from the request
        shipment_data = request.get_json()

        # Convert the shipment data to a pandas DataFrame
        df = pd.DataFrame([shipment_data])

        # Preprocess the data
        X = preprocess_data(df)

        # Make predictions using the Gradient Boosting model
        prediction = model_gradient_boosting.predict(X)[0]
        prediction_label = 'On Time' if prediction == 0 else 'Delayed'

        # Return the prediction result as JSON
        return jsonify({'prediction': prediction_label})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
