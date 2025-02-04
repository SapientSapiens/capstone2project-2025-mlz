# importing the basic  libraries
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

#loading the dataset into a dataframe
df = pd.read_csv('crop_yield.csv')

# data cleaning, preparation and feature selection

# Make column name uniform with lowercwse
df.columns = df.columns.str.lower()

# set boolean features as categorical features
df['fertilizer_used'] = df['fertilizer_used'].astype('object')
df['irrigation_used'] = df['irrigation_used'].astype('object')

# Removing data with anomalous negative value in the target feature.
df = df[df['yield_tons_per_hectare'] >= 0]

# Apply Feature Selection
categorical_columns = ['fertilizer_used', 'irrigation_used', 'soil_type']
continuous_columns = ['rainfall_mm', 'temperature_celsius']

# preparing the categorical features
dv  = DictVectorizer(sparse =False)
categorical_encoded_dict = df[categorical_columns].to_dict(orient='records')
categorical_encoded = dv.fit_transform(categorical_encoded_dict)
# convert the numpy array to a dataframe
categorical_encoded_df = pd.DataFrame(categorical_encoded, columns=dv.get_feature_names_out())

# preparing the conttinuous features
df_continuous  = df[continuous_columns]
# Removing target feature and days_to_harvest' feature before scaling as it was not included in the selected features during Feature Selection
#features = df_continuous.drop(columns=["yield_tons_per_hectare", "days_to_harvest"])
# Initialize scalers
minmax_scaler = MinMaxScaler()
# Apply MinMaxScaler 
df_minmax_scaled = pd.DataFrame(
    minmax_scaler.fit_transform(df_continuous),
    columns=df_continuous.columns
)


# concate both the categorical and continuous independent features
X = pd.concat([df_minmax_scaled, categorical_encoded_df], axis=1)
# also set the target variable
y = df['yield_tons_per_hectare']


# Selected Features for training dataset
final_selected_features = ['rainfall_mm', 'temperature_celsius', 'fertilizer_used', 'irrigation_used', 'soil_type=Chalky','soil_type=Clay', 'soil_type=Loam', 'soil_type=Peaty', 'soil_type=Sandy', 'soil_type=Silt']
feature_selected_X = X[final_selected_features]

## prepare for the train test split by joining the processed feature matrix with the target feature
final_df = feature_selected_X.copy()
final_df['yield_tons_per_hectare'] = y.values


# the actual train test split
final_X_full_train, final_X_test = train_test_split(final_df, test_size=0.2, random_state=42)

# resetting the indices of the dataframes so each starts from 0
final_X_full_train = final_X_full_train.reset_index(drop=True)
final_X_test = final_X_test.reset_index(drop=True)

# setting the target feature segment aligned to each of the fearure matrix segment
final_y_full_train  = final_X_full_train.yield_tons_per_hectare.values
y_test = final_X_test.yield_tons_per_hectare.values


# removing the target feature segment from the feature matrix segment
del final_X_full_train['yield_tons_per_hectare']
del final_X_test['yield_tons_per_hectare']

# training of the model with Linear Regression
# Initialize the Linear Regression model
lr_model = LinearRegression()
# Train the model on the training data
lr_model.fit(final_X_full_train, final_y_full_train)
# get the prediction
y_pred_lr = lr_model.predict(final_X_test)


# evaluating model with metrics
# Mean Squared Error
mse_lr = mean_squared_error(y_test, y_pred_lr)
# Root Mean Squared Error
rmse_lr = np.sqrt(mse_lr)
# R-squared
r2_lr = r2_score(y_test, y_pred_lr)
# Print Metrics
print(f'RMSE of the trained model: {rmse_lr:.4f}')
print(f'R² of the trained model: {r2_lr:.4f}')


# Save the DictVectorizer
with open('dict_vectorizer.bin', 'wb') as f_out:
    pickle.dump(dv, f_out)
print('The DictVectorizer has and saved as --> dict_vectorizer.bin')


# Save the MinMaxScaler
with open('minmax_scaler.bin', 'wb') as f_out:
    pickle.dump(minmax_scaler, f_out)
print('The MinMaxScaler has been saved as --> minmax_scaler.bin')


# saving the model
with open('model_final.bin', 'wb') as file_out:
    pickle.dump(lr_model, file_out)
file_out.close()
print('The model has been trained and saved as --> model_final.bin')
