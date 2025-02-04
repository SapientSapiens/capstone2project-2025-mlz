# importing the basic  libraries
import pandas as pd
import numpy as np
import requests

# Define the Flask API endpoint
url_predict = "http://127.0.0.1:5000/predict"  # Update this if hosted elsewhere

# Sample test data from the given dataset
test_sample_1 = {
    "Region": "North",
    "Soil_Type": "Sandy",
    "Crop": "Soybean",
    "Rainfall_mm": 986.866331,
    "Temperature_Celsius": 16.644190,
    "Fertilizer_Used": False,
    "Irrigation_Used": True,
    "Weather_Condition": "Rainy",
    "Days_to_Harvest": 146,
    "Yield_tons_per_hectare": 6.517573   # Target column to be removed
}

test_sample_2 = {
    "Region": "North",
    "Soil_Type": "Silt",
    "Crop": "Wheat",
    "Rainfall_mm": 181.587861,
    "Temperature_Celsius": 26.752729,
    "Fertilizer_Used": True,
    "Irrigation_Used": False,
    "Weather_Condition": "Sunny",
    "Days_to_Harvest": 127,
    "Yield_tons_per_hectare": 2.943716  # Target column to be removed
}

# A new hypothetical test case of unseen data not in the entire dataset
test_sample_3 = {
    "Region": "East",
    "Soil_Type": "Loam",
    "Crop": "Maize",
    "Rainfall_mm": 600.75,
    "Temperature_Celsius": 24.8,
    "Fertilizer_Used": True,
    "Irrigation_Used": True,
    "Weather_Condition": "Cloudy",
    "Days_to_Harvest": 95,
} # Yield_tons_per_hectare to be predicted 



# Remove the target feature from test samples
for sample in (test_sample_1, test_sample_2):
    sample.pop("Yield_tons_per_hectare", None)


# Handling possible exceptions and sending requests
def send_request(sample, sample_id):
    try:
        response = requests.post(url_predict, json=sample, timeout=5).json()
        predicted_yield = response["predicted_yield"]
        print(f"Test Sample {sample_id}: Predicted Yield = {predicted_yield} tons per hectare")
    except requests.exceptions.ConnectionError as e:
        print(f"Error: Could not connect to the prediction service at {url_predict}. Please ensure it is running.")
        print(f"Original error: {e}")
    except requests.exceptions.Timeout:
        print("Error: The request timed out.")
    except requests.exceptions.RequestException as e:
        print(f"Error: An unexpected error occurred: {e}")


# Send test samples for prediction
send_request(test_sample_1, 1)
send_request(test_sample_2, 2)
send_request(test_sample_3, 3)