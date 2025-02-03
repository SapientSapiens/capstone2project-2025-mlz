# Agriculture Crop Yield Predictor #

![alt text](images/modified.png)

## Problem Description ##

 **Context**

 Agriculture is one of the key pillars of the global economy, providing food and raw materials for countless industries. Accurate predictions of crop yields are crucial for ensuring food security, optimizing resource use, and supporting sustainable agricultural practices. Traditionally, predicting crop yields relied on historical data and farmer expertise, which, while valuable, often lacked the precision required for modern agricultural challenges.


 In recent years, machine learning techniques have emerged as powerful tools for enhancing yield predictions by analyzing various factors that influence crop productivity, such as weather, soil conditions, and agricultural practices. This project aims to leverage these advancements to build a robust predictive model that can estimate crop yields with high accuracy based on a range of input features.


 **The Dataset**

 The dataset is sourced from Kaggle's Agriculture Crop Yield Prediction dataset and can be found at <https://www.kaggle.com/datasets/samuelotiattakorah/agriculture-crop-yield/data>. It contains data for 1,000,000 records, each with various features related to agricultural practices and environmental conditions. The dataset is well-suited for regression tasks, with the goal of predicting the crop yield (in tons per hectare) based on factors such as weather conditions, soil type, irrigation practices, and crop variety.

 Key Attributes of the Dataset:

 - Region: The geographical region where the crop is grown (North, East, South, West).
 - Soil Type: The type of soil in which the crop is planted (Clay, Sandy, Loam, Silt, Peaty, Chalky).
 - Crop: The type of crop grown (Wheat, Rice, Maize, Barley, Soybean, Cotton).
 - Rainfall (mm): The amount of rainfall received during the crop growth period, measured in millimeters.
 - Temperature (°C): The average temperature during the crop growth period, measured in degrees Celsius.
 - Fertilizer Used: Whether fertilizer was applied during the crop growth period (True/False).
 - Irrigation Used: Whether irrigation was used during the crop growth period (True/False).
 - Weather Condition: The predominant weather condition during the crop’s growing season (Sunny, Rainy, Cloudy).
 - Days to Harvest: The number of days taken for the crop to be harvested after planting.
 - Yield (tons/hectare): The total crop yield produced, measured in tons per hectare.
 
 The data has been preprocessed to remove inconsistencies, making it ready for use in predictive modeling.


 **The Problem**

 Accurately predicting crop yield is essential for various stakeholders in the agricultural sector, including farmers, policymakers, and supply chain managers. However, this task is fraught with challenges:

 - **Environmental Variability**: Weather conditions, rainfall measure, soil types, and regional differences impact crop growth, making yield prediction highly variable.
 - **Data Inconsistencies**: Different farming practices, local environmental factors, and varying data collection standards may lead to inconsistencies in the dataset, making   predictions more complex.
 - **Resource Optimization**: In many regions, optimizing the use of resources such as irrigation and fertilizers is crucial for sustainability. Predicting the yield can help guide decisions about these resources.

 Addressing these challenges through a machine learning approach can provide more accurate and timely predictions, which in turn will help stakeholders make informed decisions about crop management, resource allocation, and planning. 


 **Solution: Project Objective**

 The objective of this project is to develop a regression model that can predict the yield of various crops based on the provided features. The model will leverage machine learning techniques to capture the complex relationships between the different factors affecting crop yield, such as measure of rainfall, agricultural practices and soil type

 The proposed model aims to achieve:

 - **High Accuracy**:  The model will use techniques such as ensemble methods (e.g., Random Forest, Gradient Boosting) and feature selection to improve accuracy and handle the complexities of the dataset.
 - **Generalizability**: The model will be trained to generalize across different input features, ensuring its adaptability to various real-world agricultural settings.
 - **Actionable Insights**: The model will provide interpretability through feature importance analysis, helping users understand the relative impact of different factors on crop yield.

 Additionally, the successful development of this crop yield prediction model can have a broad range of real-world applications:

 - **Precision Agriculture:** Helping farmers make data-driven decisions on irrigation, fertilization, and crop management to optimize yield and reduce waste.
 - **Supply Chain Management:** Supporting agricultural businesses and retailers by providing better forecasts of crop availability and helping with inventory and distribution planning.
 - **Government Planning:** Assisting policymakers in assessing agricultural productivity and formulating policies for food security and resource management.
 - **Sustainable Agriculture:** Promoting environmentally sustainable farming practices by predicting and minimizing resource usage based on crop needs and conditions.


## Exploratory Data Analysis ##


 **Cleaning and preparation of the dataset have been done as prerequisites to the EDA here involving:**

  - Loading the dataset and checking its basic information.
  - Sanitizing column names.
  - Checking null and duplicates in the dataset
  - Removing anomalous data from the target feature.
  - Assigning appropriate datatypes to featrures/columns

 For the Exploratory Data Analysis , starting with drawing inference from basic statistical data, I move on for an extensive EDA involving:

  - Analysis of the Target feature (Crop Yield in tons per hectare of land)
  - Checking count distribution and unique values of the categorical features
  - Analysis of the target variable with respect to the categorical attributes of the dataset
  - Analysis of the dataset for detecting outliers in the continuous features
  - 