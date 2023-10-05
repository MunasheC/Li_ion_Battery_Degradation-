import matplotlib.pyplot as plt
import numpy as np
from impedance.models.circuits import CustomCircuit
from impedance.visualization import plot_nyquist
from impedance.visualization import plot_bode
from impedance import preprocessing

# Read data from the text file
file_path = 'EIS_state_V_25C02.txt'
data = np.genfromtxt(file_path, delimiter='\t', skip_header=1)

# Filter data for cycle number = 1
cycle_number = data[:, 1]
filtered_data = data[cycle_number == 250]

# Extracting relevant columns for the filtered data
frequencies = filtered_data[:, 2]  # Frequency in Hz
impedance = filtered_data[:, 3] + 1j * (-filtered_data[:, 4])  # Complex impedance

#Pre-processing
# keep only the impedance data in the first quandrant
frequencies, impedance = preprocessing.ignoreBelowX(frequencies, impedance)

# Initialize Randles circuit object
circuit = 'R0-p(R1,C1)-p(R2-Ws1,C2)-p(R3,C3)'
initial_guess = [4.18e-01, 2.64e-01, 2.13e-02, 1.17e-01, 1.62e+00, 1.42e+02, 1.42e-03, 0.2, 0.03]

circuit = CustomCircuit(circuit, initial_guess=initial_guess)
# Fit the Randles model
circuit.fit(frequencies, impedance)

# Print model parameters
print(circuit)


# Predict impedance based on the model fit
impedance_fit = circuit.predict(frequencies)

# Plot the data and the fit
fig, ax = plt.subplots(figsize=(8, 5))
plot_nyquist(impedance, fmt='o', scale=10, ax=ax)
plot_nyquist(impedance_fit, fmt='-', scale=10, ax=ax)

#plot_bode(impedance, fmt='o', scale=10, ax=ax)
#//plot_bode(impedance_fit, fmt='-', scale=10, ax=ax)

plt.legend(['Data', 'Fit'])
plt.show()
