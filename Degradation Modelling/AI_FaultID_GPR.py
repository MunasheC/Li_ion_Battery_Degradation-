import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.preprocessing import StandardScaler

# Load relevant files
file_names = ['EIS_data.txt', 'Capacity_data.txt', 'EIS_data_35C02.txt', 'capacity35C02.txt']

try:
    data = [np.loadtxt(file_name) for file_name in file_names]
except:
    raise Exception('Failed to load one or more data files.')

EIS_data, Capacity_data, EIS_data_35C02, capacity35C02 = data

# Training set for the GPR model
meanEIS = np.mean(EIS_data, axis=0)
stdEIS = np.std(EIS_data, axis=0)
X_train = (EIS_data - meanEIS) / stdEIS
Y_train = Capacity_data

# Multi-Temperature EIS-Capacity GPR model
kernel = 1.0 * RBF(length_scale=1.0)
gpr = GaussianProcessRegressor(kernel=kernel, alpha=0.1, n_restarts_optimizer=10, normalize_y=True)

try:
    gpr.fit(X_train, Y_train)
except:
    raise Exception('Failed to fit the GPR model.')

# Testing set for the GPR model
X_test_35C02 = (EIS_data_35C02 - meanEIS) / stdEIS

# Capacity estimation for the testing cell
Y_test_cap_35C02, Y_test_cap_35C02_var = gpr.predict(X_test_35C02, return_std=True)

normCap35C02 = capacity35C02 / capacity35C02[0]

# Calculate the mean absolute error (MAE) between estimated and measured curves
mae = np.mean(np.abs(Y_test_cap_35C02 / Y_test_cap_35C02[0] - normCap35C02))

# Print the MAE
print('Mean Absolute Error (MAE) between estimated and measured curves:', mae)

# Plotting the estimated capacity
cycle_number = np.arange(2, 600, 2)
plt.figure()
plt.fill_between(cycle_number, (Y_test_cap_35C02 + np.sqrt(Y_test_cap_35C02_var)) / Y_test_cap_35C02[0],
                 (Y_test_cap_35C02 - np.sqrt(Y_test_cap_35C02_var)) / Y_test_cap_35C02[0], color=(255 / 255, 191 / 255, 200 / 255))
plt.plot(cycle_number, normCap35C02, 'x', color=(0 / 255, 130 / 255, 216 / 255), linewidth=3)
plt.plot(cycle_number, Y_test_cap_35C02 / Y_test_cap_35C02[0], '+', color=(205 / 255, 39 / 255, 70 / 255), linewidth=3)
plt.xlim(0, 400)
plt.ylim(0.7, 1.045)
plt.xlabel('Cycle Number', fontsize=12)
plt.ylabel('Identified Capacity', fontsize=12)
plt.title('35C02', fontsize=12)
plt.legend(['', 'Measured', 'Estimated (GPR)'], loc='upper left', bbox_to_anchor=(1, 1))

# Display the MAE on the plot
plt.text(50, 0.71, f'MAE: {mae:.4f}', fontsize=12, color='black', ha='right')

plt.show()
