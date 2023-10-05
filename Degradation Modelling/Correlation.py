import matplotlib.pyplot as plt
import numpy as np
from impedance.models.circuits import CustomCircuit
from impedance.visualization import plot_nyquist
from impedance import preprocessing

# Read data from the text files
file_path = 'EIS_state_V_45C02.txt'
file__path_cap = 'Data_Capacity_45C02.txt'
data = np.genfromtxt(file_path, delimiter='\t', skip_header=1)
data_cap = np.genfromtxt(file__path_cap, delimiter='\t', skip_header=1)

# Extract unique cycle numbers
unique_cycle_numbers = np.unique(data[:, 1])

battery_capacity = data_cap[:, 3]
unique_cycNum_cap = np.unique(data_cap[:, 1])
highest_capacity = [np.max(battery_capacity[data_cap[:, 1] == cyc]) for cyc in unique_cycNum_cap]

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
    #if i % 10 == 0:
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

highest_capacity = highest_capacity[:len(highest_capacity)-1]
proc_resistances_R0 = resistances_R0[:len(highest_capacity)]
proc_resistances_R1 = resistances_R1[:len(highest_capacity)]
proc_resistances_R2 = resistances_R2[:len(highest_capacity)]
proc_WsMag = WsMag[:len(highest_capacity)]
proc_WsTau = WsTau[:len(highest_capacity)]
proc_capacitances_C1 = capacitances_C1[:len(highest_capacity)]
proc_capacitances_C2 = capacitances_C2[:len(highest_capacity)]

#print(len(proc_resistances_R0))
#print(len(highest_capacity))

# Plot the resistances, capacitances, and Wo parameters vs. cycle number
correlation_coefficient_R0 = np.corrcoef(proc_resistances_R0, highest_capacity)[0, 1]
correlation_coefficient_R1 = np.corrcoef(proc_resistances_R1, highest_capacity)[0, 1]
correlation_coefficient_R2 = np.corrcoef(proc_resistances_R2, highest_capacity)[0, 1]
correlation_coefficient_WsMag = np.corrcoef(proc_WsMag, highest_capacity)[0, 1]
correlation_coefficient_WsTau = np.corrcoef(proc_WsTau, highest_capacity)[0, 1]
correlation_coefficient_C1 = np.corrcoef(proc_capacitances_C1, highest_capacity)[0, 1]
correlation_coefficient_C2 = np.corrcoef(proc_capacitances_C2, highest_capacity)[0, 1]

print("The correlation coefficient of bulk resistance and battery capacity is: ", correlation_coefficient_R0)
print("The correlation coefficient of interface layer resistance and battery capacity is: ", correlation_coefficient_R1)
print("The correlation coefficient of charge transfer resistance and battery capacity is: ", correlation_coefficient_R2)
print("The correlation coefficient of Warburg magnitude and battery capacity is: ", correlation_coefficient_WsMag)
print("The correlation coefficient of Warburg tau and battery capacity is: ", correlation_coefficient_WsTau)
print("The correlation coefficient of interface layer capacitance and battery capacity is: ", correlation_coefficient_C1)
print("The correlation coefficient of double layer capacitance and battery capacity is: ", correlation_coefficient_C2)