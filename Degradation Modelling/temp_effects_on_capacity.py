import numpy as np
import matplotlib.pyplot as plt

# Read data from a text file and keep only highest capacity for each cycle number
def read_data(filename):
    data = np.loadtxt(filename, delimiter='\t', skiprows=1)
    
    # Extract cycle number and capacity
    if filename == 'Data_Capacity_25C02.txt':
        cyc_num = data[:,1]
        capacity = data[:,5]
    else: 
        cyc_num = data[:, 1]
        capacity = data[:, 3]  # Capacity column
    
    # Keep only the highest capacity for each unique cycle number
    unique_cycles = np.unique(cyc_num)
    highest_capacity = [np.max(capacity[cyc_num == cyc]) for cyc in unique_cycles]
    
    return unique_cycles, highest_capacity

# Plot data and trendline for a given temperature
def plot_data_and_trendline(ax, cyc_num, capacity, temperature):
    ax.plot(cyc_num, capacity, 'b-', linewidth=2, label='Highest Capacity')
    ax.set_xlabel('Cycle number')
    ax.set_ylabel('Capacity (mA.h)')
    ax.set_title(f'Highest Capacity at {temperature} degrees Celsius')
    ax.grid(True)
    ax.legend()

# Load data for each temperature
cyc_num25, capacity25 = read_data('Data_Capacity_25C02.txt')
cyc_num35, capacity35 = read_data('Data_Capacity_35C02.txt')
cyc_num45, capacity45 = read_data('Data_Capacity_45C02.txt')

# Determine common axis limits
x_min = min(np.concatenate((cyc_num25, cyc_num35, cyc_num45)))
x_max = max(np.concatenate((cyc_num25, cyc_num35, cyc_num45)))
y_min = min(np.concatenate((capacity25, capacity35, capacity45)))
y_max = max(np.concatenate((capacity25, capacity35, capacity45)))

# Create subplots
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot for 25 degrees Celsius
plot_data_and_trendline(axes[0], cyc_num25, capacity25, 25)

# Plot for 35 degrees Celsius
plot_data_and_trendline(axes[1], cyc_num35, capacity35, 35)

# Plot for 45 degrees Celsius
plot_data_and_trendline(axes[2], cyc_num45, capacity45, 45)

# Set common axis limits
for ax in axes:
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])

plt.tight_layout()
plt.show()
