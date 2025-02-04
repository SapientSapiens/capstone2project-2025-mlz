import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify


# Load the trained linear regression model
with open('dict_vectorizer.bin', 'rb') as f_in:
    dv = pickle.load(f_in)
    print('dict_vectorizer loaded..!')
    
with open('minmax_scaler.bin', 'rb') as f_in:
    minmax_scaler = pickle.load(f_in)
    print('minmax_scaler loaded..!')

model_file = 'model_final.bin'
with open(model_file, 'rb') as f_in:
    model = pickle.load(f_in)
    print('Crop Yield Predictor Model deployed..!')

# Initialize Flask app
app = Flask('crop_yield_predictor')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input data (assumed to be JSON)
        data = request.get_json()
        print("Received request data:", data)
        print("hello.......")
        # Convert the input data into a DataFrame
        df_test = pd.DataFrame([data])
        print("Received --------------:", df_test.shape)
        
        # Make column name uniform with lowercwse
        df_test.columns = df_test.columns.str.lower()
        print("Columns --------------:", df_test.columns)
        # set boolean features as categorical features
        df_test['fertilizer_used'] = df_test['fertilizer_used'].astype('object')
        df_test['irrigation_used'] = df_test['irrigation_used'].astype('object')
        print("debug 1")
        # Selecting only the relevant columns (same as during training)
        categorical_columns = ['fertilizer_used', 'irrigation_used', 'soil_type'] # Categorical features
        continuous_columns = ['rainfall_mm', 'temperature_celsius']  # Continuous features
        print("debug 2")
        # Extract data for categorical and continuous columns
        df_test_categorical = df_test[categorical_columns]
        df_test_continuous = df_test[continuous_columns]

        print("debug 3")
        # One-hot encode categorical columns using the same DictVectorizer from training
        categorical_encoded = dv.transform(df_test_categorical.to_dict(orient='records'))
        categorical_encoded_df = pd.DataFrame(categorical_encoded, columns=dv.get_feature_names_out())

        print("debug 4", df_test_continuous.columns)
        # Scale the continuous columns using the same MinMaxScaler from training
        df_test_continuous_scaled = pd.DataFrame(minmax_scaler.transform(df_test_continuous), columns=df_test_continuous.columns)
        print("debug 5")
        # Concatenate the encoded categorical and scaled continuous features
        df_test_final = pd.concat([df_test_continuous_scaled, categorical_encoded_df], axis=1)
        print("debug a: categorical_encoded_df columns:", categorical_encoded_df.columns)
        print("debug b: df_test_continuous_scaled columns:", df_test_continuous_scaled.columns)
        print("debug 6", df_test_final.columns)
        # Predict using the model
        y_pred = model.predict(df_test_final)[0]  # Predict yield
        y_pred_rounded = round(y_pred, 4)
        print("debug 7")
        # Prepare the result
        result = {
            'predicted_yield': float(y_pred_rounded)
        }
        print("debug 8", result)
        # Return the result as a JSON response
        return jsonify(result), 200
    

    except Exception as e:
        return jsonify({'error': str(e)}), 400



if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
