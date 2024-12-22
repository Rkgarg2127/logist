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

def preprocess_data(data):
    # Convert categorical columns to dummy variables
    categorical_columns = ['Vehicle Type', 'Weather Conditions', 'Traffic Conditions', 'Origin', 'Destination']
    data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)
    
    # Retrieve the correct column names for the model (the ones it was trained on)
    model_columns = ['Distance (km)', 'Vehicle Type_Lorry', 'Vehicle Type_Trailer', 'Vehicle Type_Truck', 'Weather Conditions_Fog', 'Weather Conditions_Rain', 'Weather Conditions_Storm', 'Traffic Conditions_Light', 'Traffic Conditions_Moderate', 'Origin_Bangalore', 'Origin_Chennai', 'Origin_Delhi', 'Origin_Hyderabad', 'Origin_Jaipur', 'Origin_Kolkata', 'Origin_Lucknow', 'Origin_Mumbai', 'Origin_Pune', 'Destination_Bangalore', 'Destination_Chennai', 'Destination_Delhi', 'Destination_Hyderabad', 'Destination_Jaipur', 'Destination_Kolkata', 'Destination_Lucknow', 'Destination_Mumbai', 'Destination_Pune']
    
    current_columns = data.columns.tolist()

    # Find missing columns (columns expected by the model but not present in the incoming data)
    missing_columns = set(model_columns) - set(current_columns)
    
    # Add missing columns with value 0 (to match the model's expected input)
    for col in missing_columns:
        data[col] = 0

    # Reorder columns to match the model's expected input
    data = data[model_columns]  # Reorder to match model's input
    
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
