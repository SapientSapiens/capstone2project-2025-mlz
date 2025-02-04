import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

# Define the features to keep
keys_to_keep = [
    'rainfall_mm', 'fertilizer_used', 'irrigation_used', 'temperature_celsius',
    'soil_type=Clay', 'soil_type=Loam', 'soil_type=Sandy', 'soil_type=Chalky',
    'soil_type=Peaty', 'soil_type=Silt'
]

# Load the trained linear regression model
model_file = 'model_final.bin'
with open(model_file, 'rb') as f_in:
    model = pickle.load(f_in)
    print('Crop Yield Predictor Model deployed..!')


# Initialize Flask app
app = Flask('crop_yield_predictor')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    print("Received request data:", data)
    
    # Extract only relevant features
    filtered_data = {key: data.get(key, 0) for key in keys_to_keep}  # Default missing keys to 0
    df_test = pd.DataFrame([filtered_data])
    print("Extracted only relevant feature: ", df_test.columns)

    # Predict using the model
    y_pred = model.predict(df_test)[0]  # Predict yield
    y_pred_rounded = round(y_pred, 4)
    
    result = {
        'predicted_yield': float(y_pred_rounded)
    }
    
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
