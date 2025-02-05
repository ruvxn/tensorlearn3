# Description: This file is used to load the dataset and clean the data
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Load the data
bike_data = pd.read_csv('dataset/bike_sharing_daily.csv')

#identify the data
print(bike_data.head(10))

sns.heatmap(bike_data.isnull())
plt.show()

# Clean the data
bike_data=bike_data.drop(labels=['instant', 'casual','registered'],axis=1)

# Fix the date format
bike_data['dteday'] = pd.to_datetime(bike_data['dteday'], format='%m/%d/%Y')

# Verify changes
print(bike_data.head(10))

#Change the index to the date
bike_data.index = pd.DatetimeIndex(bike_data['dteday'])
bike_data.drop('dteday', axis=1, inplace=True)

# Verify changes
print(bike_data.head(10))


# Plot the count of bikes rented per week
bike_data['cnt'].asfreq('W').plot(linewidth=3)
plt.title('Bikes Rented Weekly')
plt.xlabel('Week')
plt.ylabel('Bikes Rented')
plt.show()

# Plot the count of bikes rented per month
bike_data['cnt'].asfreq('M').plot(linewidth=3)
plt.title('Bikes Rented Monthly')
plt.xlabel('Month')
plt.ylabel('Bikes Rented')
plt.show()

# Plot the count of bikes rented per quarter

bike_data['cnt'].asfreq('QE').plot(linewidth=3)
plt.title('Bikes Rented Quarterly')
plt.xlabel('Quarter')
plt.ylabel('Bikes Rented')
plt.show()

# Plot the correlation between the numerical features
X_numerical = bike_data[['temp', 'hum', 'windspeed','cnt']]
sns.heatmap(X_numerical.corr(), annot=True)
plt.show()

#save the cleaned data
bike_data.to_csv('dataset/bike_rental_cleaned.csv')
print('Data cleaned and saved')