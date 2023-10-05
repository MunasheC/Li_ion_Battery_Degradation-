import matplotlib.pyplot as plt
import numpy as np
from impedance.models.circuits import CustomCircuit
from impedance.visualization import plot_nyquist
from impedance import preprocessing

# Read data from the text files
file_path = 'EIS_state_IX_45C02.txt'
file__path_cap = 'Data_Capacity_25C02.txt'
data = np.genfromtxt(file_path, delimiter='\t', skip_header=1)
data_cap = np.genfromtxt(file__path_cap, delimiter='\t', skip_header=1)

# Extract unique cycle numbers
unique_cycle_numbers = np.unique(data[:, 1])

# Initialize empty lists to store parameters
resistances_R0 = []
resistances_R1 = []
resistances_R2 = []
WsMag = []
WsTau = []
capacitances_C1 = []
capacitances_C2 = []

# Loop through unique cycle numbers and extract parameters for every 10th cycle
for i, cycle_number in enumerate(unique_cycle_numbers):
    if i % 10 == 0:
        filtered_data = data[data[:, 1] == cycle_number]

        # Extracting relevant columns for the filtered data
        frequencies = filtered_data[:, 2]  # Frequency in Hz
        impedance = filtered_data[:, 3] + 1j * (-filtered_data[:, 4])  # Complex impedance

        # Pre-processing: keep only the impedance data in the first quadrant
        frequencies, impedance = preprocessing.ignoreBelowX(frequencies, impedance)

        # Initialize Randles circuit object
        circuit = 'R0-p(R1,C1)-p(R2-Ws1,C2)'
        initial_guess = [4.18e-01, 2.64e-01, 2.13e-02, 1.17e-01, 1.62e+00, 1.42e+02, 1.42e-03]

        circuit = CustomCircuit(circuit, initial_guess=initial_guess)
        # Fit the Randles model
        circuit.fit(frequencies, impedance)

        # Append parameters to respective lists
        resistances_R0.append(circuit.parameters_[0])
        resistances_R1.append(circuit.parameters_[1])
        resistances_R2.append(circuit.parameters_[3])
        WsMag.append(circuit.parameters_[4])
        WsTau.append(circuit.parameters_[5])
        capacitances_C1.append(circuit.parameters_[2])
        capacitances_C2.append(circuit.parameters_[6])

# Plot the resistances, capacitances, and Wo parameters vs. cycle number
plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.plot(np.arange(0, len(resistances_R0) * 10, 10), resistances_R0, 'o-')
plt.xlabel('Cycle Number')
plt.ylabel('Resistance R0 (Ohm)')
plt.title('Solution Resistance vs. Cycle Number')

plt.subplot(2, 3, 2)
plt.plot(np.arange(0, len(resistances_R1) * 10, 10), resistances_R1, 'o-')
plt.xlabel('Cycle Number')
plt.ylabel('Resistance R1 (Ohm)')
plt.title('Interface Layer Resistance vs. Cycle Number')

plt.subplot(2, 3, 3)
plt.plot(np.arange(0, len(resistances_R2) * 10, 10), resistances_R2, 'o-')
plt.xlabel('Cycle Number')
plt.ylabel('Resistance R2 (Ohm)')
plt.title('Charge Transfer Resistance vs. Cycle Number')

plt.subplot(2, 3, 4)
plt.plot(np.arange(0, len(capacitances_C1) * 10, 10), capacitances_C1, 'o-')
plt.plot(np.arange(0, len(capacitances_C2) * 10, 10), capacitances_C2, 'o-')
plt.xlabel('Cycle Number')
plt.ylabel('Capacitance (F)')
plt.legend(['C1', 'C2'])
plt.title('Capacitances vs. Cycle Number')

plt.subplot(2, 3, 5)
plt.plot(np.arange(0, len(WsMag) * 10, 10), WsMag, 'o-')
plt.xlabel('Cycle Number')
plt.ylabel('Magnitude (Ohm)')
plt.title('Warburg Magnitude vs. Cycle Number')

plt.subplot(2, 3, 6)
plt.plot(np.arange(0, len(WsTau) * 10, 10), WsTau, 'o-')
plt.xlabel('Cycle Number')
plt.ylabel('Tau (s)')
plt.title('Warburg Tau vs. Cycle Number')



plt.tight_layout()
plt.show()
