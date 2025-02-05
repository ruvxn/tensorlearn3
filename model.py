import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score  # ✅ Import Fixed

# Load the data
bike_data = pd.read_csv('dataset/bike_rental_cleaned.csv')

# Extract the features and target
X = bike_data.drop(columns=['cnt'])
y = bike_data['cnt']

# Scale the target variable (y) to range [0,1] using MinMaxScaler
scaler = MinMaxScaler()
y = scaler.fit_transform(y.values.reshape(-1, 1))  # ✅ Ensure y is 2D for scaling

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the neural network model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='linear')  # Linear output for regression
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(X_train, y_train, epochs=200, batch_size=64, validation_split=0.2)

#save the model
model.save("saved_model/bike_rental_model.h5")  
print("Model saved successfully as 'bike_rental_model.h5'")

# Plot Training & Validation Loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss Progress during Training')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Make predictions on the test set
predictions = model.predict(X_test)

# Convert scaled predictions back to original scale
y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1))  # Convert back to original scale
predictions_original = scaler.inverse_transform(predictions)       # Convert back to original scale

# Calculate evaluation metrics
mse = mean_squared_error(y_test_original, predictions_original)  # ✅ Fixed Import
rmse = np.sqrt(mse)  # Root Mean Squared Error
mae = mean_absolute_error(y_test_original, predictions_original)
r2 = r2_score(y_test_original, predictions_original)

# Print evaluation metrics
print(f'Mean Squared Error (MSE): {mse:.4f}')
print(f'  ▶ Measures the **average squared difference** between actual and predicted values.')
print(f'  ▶ A lower value is better (closer to 0).')

print(f'Root Mean Squared Error (RMSE): {rmse:.4f}')
print(f'  ▶ Measures the **spread of errors** in the same units as the target variable.')
print(f'  ▶ A lower RMSE means better accuracy.')

print(f'Mean Absolute Error (MAE): {mae:.4f}')
print(f'  ▶ Measures the **average absolute difference** between predicted and actual values.')
print(f'  ▶ A lower MAE means the model makes smaller errors on average.')

print(f'R² Score: {r2:.4f}')
print(f'  ▶ Indicates how well the model explains the variance in the data.')
print(f'  ▶ R² = 1 means **perfect prediction**, R² = 0 means **no predictive power**.')
print(f'  ▶ Higher is better (close to 1.0).')

# Scatter plot: True vs Predicted Values
plt.scatter(y_test_original, predictions_original, alpha=0.5)
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('True vs Predicted Values')
plt.show()

