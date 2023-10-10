import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense


# Load relevant files
file_names = ['EIS_data.txt', 'Capacity_data.txt', 'EIS_data_35C02.txt', 'capacity35C02.txt']

try:
    data = [np.loadtxt(file_name) for file_name in file_names]
except:
    raise Exception('Failed to load one or more data files.')

EIS_data, Capacity_data, EIS_data_35C02, capacity35C02 = data

# Training set for the neural network
meanEIS = np.mean(EIS_data, axis=0)
stdEIS = np.std(EIS_data, axis=0)
X_train = (EIS_data - meanEIS) / stdEIS
Y_train = Capacity_data

# Split the data into training and testing sets
X_train, X_val, y_train, y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)

# Build a feedforward neural network model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1)  # Output layer with linear activation for regression
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=16, verbose=1)

# Testing set for the neural network
X_test_35C02 = (EIS_data_35C02 - meanEIS) / stdEIS

# Predict capacity for the testing set
Y_test_cap_35C02_nn = model.predict(X_test_35C02).flatten()

normCap35C02 = capacity35C02 / capacity35C02[0]
# Calculate the mean absolute error (MAE) between estimated and measured curves
mae_nn = mean_absolute_error(Y_test_cap_35C02_nn / Y_test_cap_35C02_nn[0], Y_test_cap_35C02_nn)
mae_nn = np.mean(np.abs(Y_test_cap_35C02_nn / Y_test_cap_35C02_nn[0] - normCap35C02))


# Print the MAE
print('Mean Absolute Error (MAE) between estimated and measured curves using a neural network:', mae_nn)

# Plotting the estimated capacity
cycle_number = np.arange(2, 600, 2)
plt.figure()
plt.fill_between(cycle_number, Y_test_cap_35C02_nn, color=(255 / 255, 191 / 255, 200 / 255))
plt.plot(cycle_number, capacity35C02 / capacity35C02[0], 'x', color=(0 / 255, 130 / 255, 216 / 255), linewidth=3)
plt.plot(cycle_number, Y_test_cap_35C02_nn / Y_test_cap_35C02_nn[0], '+', color=(205 / 255, 39 / 255, 70 / 255), linewidth=3)
plt.xlim(0, 400)
plt.ylim(0.7, 1.045)
plt.xlabel('Cycle Number', fontsize=12)
plt.ylabel('Identified Capacity', fontsize=12)
plt.title('35C02', fontsize=12)
plt.legend(['', 'Measured', 'Estimated (NN)'], loc='upper left', bbox_to_anchor=(1, 1))

# Display the MAE on the plot
plt.text(50, 0.71, f'MAE: {mae_nn:.4f}', fontsize=12, color='black', ha='right')

plt.show()
