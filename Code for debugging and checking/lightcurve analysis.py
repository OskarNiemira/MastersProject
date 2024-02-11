#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

A code for analyzing RXTE light curve data for thermonuclear bursts in LMXBs
"""
import astropy.io.fits as fits
import pandas as pd
import matplotlib.pyplot as plt
import os
from astropy.stats import sigma_clip
from astropy.time import Time
import numpy as np


directory_path = r"C:\Users\oskar\OneDrive - University of Southampton\Desktop\physics\year 4\MastersProject\lokalne testy\data\FS46_16e96bb0-16e9735c_std1.lc"

# FITS file name
fits_file = "std1.lc"

if os.path.isfile(directory_path):
    print(f"The file exists at {directory_path}")
else:
    print(f"The file does not exist at {directory_path}. Please check the path.")

# Full path to the FITS file
#full_path = os.path.join(directory_path, fits_file)
full_path = directory_path
# Assert that the full path points to a file
assert os.path.isfile(full_path)

# Load the RXTE data from a FITS file
hdul = fits.open(full_path)
# Print the HDU (Header Data Unit) information
hdul.info()

# Access the RATE HDU
rate_hdu = hdul['RATE']

# Extract the light curve data
data = rate_hdu.data

print(data.columns.names)
print(hdul)
print(hdul['RATE'].columns.names)
time_column = data['TIME']

flux_data = data['RATE']


time_values = time_column.tolist()  # Convert Time object to a list of MJD values

"""
# Background Subtraction
# Calculate the sigma-clipped background
background_level = sigma_clip(flux_data, sigma=3).mean()

# Subtract the background level from the flux data
background_subtracted_flux = flux_data - background_level
"""

# Burst Detection
# Define the threshold value for burst detection
burst_threshold = np.median(flux_data) * 1.5

def threshold_crossing_detection(flux, threshold):
    """
    Detects threshold crossings in the flux data to identify burst start and end times.

    Parameters:
    flux (array): The flux data.
    threshold (float): The threshold value for detecting bursts.

    Returns:
    tuple of arrays: Arrays of burst start times and end times indices.
    """
    above_threshold = flux > threshold
    start_times = []
    end_times = []

    for i in range(1, len(flux)):
        if above_threshold[i] and not above_threshold[i - 1]:
            start_times.append(i)
        elif not above_threshold[i] and above_threshold[i - 1]:
            end_times.append(i)

    # Ensure that every start has an end
    if len(end_times) < len(start_times):
        end_times.append(len(flux) - 1)

    return np.array(start_times), np.array(end_times)

# Detect bursts using threshold crossing detection
burst_indices_start, burst_indices_end = threshold_crossing_detection(flux_data, burst_threshold)

# Convert indices to time values
burst_start_times = [time_values[i] for i in burst_indices_start]
burst_end_times = [time_values[i] for i in burst_indices_end]

# Burst Duration Calculation
# Calculate the duration of each detected burst
burst_durations = [end - start for start, end in zip(burst_start_times, burst_end_times)]

# Plotting
plt.figure(figsize=(10, 6))

# Plot the background-subtracted light curve
plt.scatter(time_values, flux_data, s=5, marker='.', color='blue')


# Mark burst start and end times
for burst_start in burst_start_times:
    plt.axvline(x=burst_start, color='green', linestyle='--', label='Burst Start' if burst_start == burst_start_times[0] else "")

for burst_end in burst_end_times:
    plt.axvline(x=burst_end, color='red', linestyle=':', label='Burst End' if burst_end == burst_end_times[0] else "")


# Mark burst start and end times
for burst_start, burst_end in zip(burst_start_times, burst_end_times):
    plt.axvspan(burst_start, burst_end, color='orange', alpha=0.3)


# Add labels and title
plt.xlabel('Time ')
plt.ylabel('Flux')
plt.title('RXTE Light Curve with Burst Detection')
plt.show()

# Directory to save burst files
burst_save_directory = r"C:\Users\oskar\OneDrive - University of Southampton\Desktop\physics\year 4\MastersProject\MastersProject\tests"
  
for i, (start, end) in enumerate(zip(burst_indices_start, burst_indices_end)):
    burst_data = rate_hdu.data[start:end]  # Access the raw array from the RATE HDU
    burst_df = pd.DataFrame(burst_data)
    burst_file_name = f"burst_{i}.csv"
    burst_file_path = os.path.join(burst_save_directory, burst_file_name)
    burst_df.to_csv(burst_file_path, index=False)


hdul.close()