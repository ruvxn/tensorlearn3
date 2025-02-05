# Description: This file is used to load the dataset and clean the data
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder

# Load the data
bike_data = pd.read_csv('dataset/bike_sharing_daily.csv')

# Identify the data
print(bike_data.head(10))

# Check for missing values
plt.figure(figsize=(10, 6))
sns.heatmap(bike_data.isnull(), cmap="viridis", cbar=False)
plt.show()

# Clean the data: Drop unnecessary columns
bike_data = bike_data.drop(columns=['instant', 'casual', 'registered'])

# Fix the date format (automatically detects format)
bike_data['dteday'] = pd.to_datetime(bike_data['dteday'], infer_datetime_format=True)

# Verify changes
print(bike_data.head(10))

# Change the index to the date
bike_data = bike_data.set_index('dteday')

# Verify changes
print(bike_data.head(10))

# Plot the count of bikes rented per week
bike_data['cnt'].resample('W').sum().plot(linewidth=3)
plt.title('Bikes Rented Weekly')
plt.xlabel('Week')
plt.ylabel('Bikes Rented')
plt.show()

# Plot the count of bikes rented per month
bike_data['cnt'].resample('M').sum().plot(linewidth=3)
plt.title('Bikes Rented Monthly')
plt.xlabel('Month')
plt.ylabel('Bikes Rented')
plt.show()

# Plot the count of bikes rented per quarter
bike_data['cnt'].resample('Q').sum().plot(linewidth=3)
plt.title('Bikes Rented Quarterly')
plt.xlabel('Quarter')
plt.ylabel('Bikes Rented')
plt.show()

# Plot the correlation between the numerical features
X_numerical = bike_data[['temp', 'hum', 'windspeed', 'cnt']]
sns.heatmap(X_numerical.corr(), annot=True, cmap="coolwarm")
plt.show()

# Get the categorical features
X_categorical = bike_data[['season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit']]

# One-hot encode categorical features with meaningful column names
encoder = OneHotEncoder(sparse_output=False)
X_categorical_encoded = encoder.fit_transform(X_categorical)

# Retrieve column names for one-hot encoding
encoded_columns = encoder.get_feature_names_out(X_categorical.columns)

# Convert the encoded array back to DataFrame with proper column names
X_categorical_encoded = pd.DataFrame(X_categorical_encoded, columns=encoded_columns)

# Reset index for numerical features
X_numerical = X_numerical.reset_index(drop=True)
X_categorical_encoded = X_categorical_encoded.reset_index(drop=True)

# Combine numerical and categorical features
X_all = pd.concat([X_categorical_encoded, X_numerical], axis=1)

# Display summary to check for NaN values
print(X_all)

# Save the cleaned data
X_all.to_csv('dataset/bike_rental_cleaned.csv', index=False)
print('Data cleaned and saved successfully!')
