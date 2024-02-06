#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A code for analyzing RXTE light curve data for thermonuclear bursts in LMXBs
"""
import astropy.io.fits as fits
import os
import numpy as np
import sys

# Verify that two arguments are passed (script name, file path, and output directory)
if len(sys.argv) != 3:
    print("Usage: {} <input file> <output directory>".format(sys.argv[0]))
    sys.exit(1)

file_path = sys.argv[1]
output_directory = sys.argv[2]

try:
    # Load the RXTE data from the FITS file
    with fits.open(file_path) as hdul:
        data = hdul['RATE'].data
        time_column = data['TIME']
        flux_data = data['RATE']
    print("Data loaded successfully from {}".format(file_path))
except Exception as e:
    print("Failed to load data: {}".format(e))
    sys.exit(1)
#end of check point
"""
# Assuming the first argument is the file path
file_path = sys.argv[1]
output_directory = sys.argv[2]  # Assuming the second argument is the output directory

# Load the RXTE data from the FITS file
with fits.open(file_path) as hdul:
    data = hdul['RATE'].data
    time_column = data['TIME']
    flux_data = data['RATE']
"""
# Convert Time object to a list of MJD values
time_values = time_column.tolist()

# Burst Detection
burst_threshold = np.median(flux_data) * 1.1

def threshold_crossing_detection(flux, threshold):
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

burst_indices_start, burst_indices_end = threshold_crossing_detection(flux_data, burst_threshold)

burst_detected = True  # Placeholder; set this based on your burst detection logic
if burst_detected:
    # Assuming burst_indices_start and burst_indices_end define the burst data range
    # For demonstration, let's assume they are defined
    burst_data = data[burst_indices_start:burst_indices_end]
    
    # Create a new FITS file with the burst data
    hdu = fits.BinTableHDU(data=burst_data)
    burst_file_name = os.path.basename(file_path).replace('.fits', '_burst.fits')
    burst_file_path = os.path.join(output_directory, burst_file_name)
    
    hdu.writeto(burst_file_path, overwrite=True)

# No need to save anything if no burst is detected, the script will simply end