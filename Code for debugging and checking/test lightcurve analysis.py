#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A code for analyzing RXTE light curve data for thermonuclear bursts in LMXBs
"""
import astropy.io.fits as fits
import os
import numpy as np
import sys
import matplotlib.pyplot as plt
import pandas as pd  # Import pandas for rolling statistics
# Define the file path and output directory directly
file_path = r"/Users/oskarniemira/Desktop/Masters Project/lokalne testy/data/He bursts/data_4u1728/full_data/FS46_9b61db5-9b62a00_std1.lc"
#output_directory = r"C:\Users\oskar\OneDrive - University of Southampton\Desktop\physics\year 4\MastersProject\lokalne testy\outputs"

try:
    # Load the RXTE data from the FITS file
    with fits.open(file_path) as hdul:
        data = hdul['RATE'].data
        time_column = data['TIME']
        flux_data = data['RATE']
        error_data = data['ERROR']
    print("Data loaded successfully from {}".format(file_path))

    # Plot the original light curve
    plt.figure(figsize=(10, 4))
    plt.plot(time_column, flux_data, label='Original Data')
    plt.xlabel('Time (MJD)')
    plt.ylabel('Flux')
    #plt.yscale('log')
    plt.title('Original Light Curve')
    plt.legend()
    plt.show()
except Exception as e:
    print("Failed to load data: {}".format(e))
    sys.exit(1)

# Calculate mean and standard deviation excluding zero values (if zeros indicate no observation)
valid_flux_data = flux_data[flux_data > 200]
mean_flux = np.mean(valid_flux_data)
std_flux = np.std(valid_flux_data)
median = np.median(valid_flux_data)
print("Mean: ", mean_flux)
print("Std: ", std_flux)
print("Median: ", median)
# Define the burst threshold as 4 times the standard deviation above the mean
burst_threshold = mean_flux + 4 * std_flux
print("Burst detection threshold:", burst_threshold)
absolute_threshold = 1000

def threshold_crossing_detection(flux, threshold, absolute_threshold):
   above_threshold = flux > threshold
   above_absolute_threshold = flux > absolute_threshold
   start_times = []
   end_times = []

   for i in range(1, len(flux)):
        if above_threshold[i] and above_absolute_threshold[i] and not above_threshold[i - 1]:
            start_times.append(i)
        elif not (above_threshold[i] and above_absolute_threshold[i]) and above_threshold[i - 1]:
            end_times.append(i)

   # Ensure that every start has an end
   if len(end_times) < len(start_times):
       end_times.append(len(flux) - 1)

   return np.array(start_times), np.array(end_times)

burst_indices_start, burst_indices_end = threshold_crossing_detection(flux_data, burst_threshold, absolute_threshold)
print(burst_indices_start,burst_indices_end)

def find_initial_growing_phase(flux, start_times, std_flux, lookback=100):
    """
    Adjusts burst start times to the earliest point in the growing phase where 
    the flux increase is greater than a threshold.

    :param flux: The flux data array.
    :param start_times: The initially detected start times of bursts.
    :param std_flux: Standard deviation of flux used to calculate the threshold.
    :param lookback: The maximum number of points to look back for the growing phase.
    :return: Adjusted start times including the earliest point in the growing phase.
    """
    adjusted_start_times = []

    # Check if the flux array and start_times list are not empty
    if flux.size == 0 or len(start_times) == 0:
        return adjusted_start_times  # Return an empty list if there's nothing to process

    threshold_point = std_flux * 2

    for start in start_times:
        if start >= len(flux):
            # If the start index is out of bounds, skip this start time
            continue

        # Find the earliest start point where the increase is above the threshold
        while start > 0 and (flux[start] - flux[start - 1]) > threshold_point:
            start -= 1

        # The current 'start' is the last index where the increase was above the threshold,
        # so the actual start might be the next index
        adjusted_start = max(start - 2, 0)

        # Ensure we don't go beyond the array bounds
        adjusted_start_times.append(adjusted_start)

    return adjusted_start_times


# Use the function to adjust start times
adjusted_start_times = find_initial_growing_phase(flux_data, burst_indices_start, std_flux)

def adjust_finish_times(flux, end_times, mean_flux, std_flux, window_size=40, threshold=0.2):
    """
    Adjusts burst end times based on when the flux stabilizes within a specified range of the mean.

    :param flux: The flux data array.
    :param end_times: The initially detected end times of bursts.
    :param mean_flux: Mean flux calculated from non-burst periods.
    :param std_flux: Standard deviation of flux from non-burst periods (not used in this implementation).
    :param window_size: The number of points to average over to determine the post-burst flux level.
    :param threshold: The allowed deviation from the mean flux as a fraction (e.g., 0.1 for 10%).
    :return: Adjusted end times to include the gradual drop-off phase.
    """
    adjusted_end_times = []
    allowed_deviation = mean_flux * threshold

    for end in end_times:
            # Initialize potential end as the first point after the burst end
            potential_end = end + 1
    
            # Loop over each point after the burst end to find where the flux returns to the mean
            while potential_end < len(flux):
                window_start = max(0, potential_end - window_size)
                window_flux = flux[window_start:potential_end]
                avg_window_flux = np.mean(window_flux)
    
                if abs(avg_window_flux - mean_flux) <= allowed_deviation:
                    # If the average window flux is within the allowed deviation,
                    # this is the new adjusted end time
                    adjusted_end = potential_end
                    break
    
                potential_end += 1  # Increment to check the next point
    
            # If we didn't find a point where flux returns to the mean,
            # set the adjusted end time to the last checked point
            else:
                adjusted_end = potential_end
    
            # Append the found or last checked point to the adjusted_end_times list
            adjusted_end_times.append(adjusted_end)
    
    return adjusted_end_times

adjusted_end_times = adjust_finish_times(flux_data, burst_indices_end, mean_flux, std_flux)

print("adjusted bursts start: ", adjusted_start_times)
print("adjusted bursts end: ", adjusted_end_times)

def filter_bursts_by_flux_change(time_column, flux_data, start_indices, end_indices, window_size=1500, threshold=0.2):
    # Initialize lists to hold the indices of bursts that are not significantly different
    filtered_start_indices = []
    filtered_end_indices = []
        
    # Iterate over each burst to evaluate the mean flux before and after
    for start, end in zip(start_indices, end_indices):
        # Calculate pre-burst mean
        pre_window_start = max(0, start - window_size)
        pre_window_end = max(0, start)
        pre_burst_mean = np.mean(flux_data[pre_window_start:pre_window_end])
        print('pre mean ',pre_burst_mean)
        
        # Calculate post-burst mean
        post_window_start = min(len(flux_data) - 1, end)
        post_window_end = min(len(flux_data) - 1, end + window_size)
        post_burst_mean = np.mean(flux_data[post_window_start:post_window_end])
        print('post mean ',post_burst_mean)

        # Determine if the difference is significant
        if pre_burst_mean == 0: # Avoid division by zero
            relative_change = 0
        else:
            relative_change = abs(pre_burst_mean - post_burst_mean) / pre_burst_mean
            print('rela: ',relative_change)
        # Append indices if the change is not significant
        if relative_change <= threshold:
            filtered_start_indices.append(start)
            filtered_end_indices.append(end)
        
    # Return the filtered indices
    return filtered_start_indices, filtered_end_indices

filtered_start_times, filtered_end_times = filter_bursts_by_flux_change(time_column, flux_data, adjusted_start_times, adjusted_end_times)

if len(filtered_start_times) > 0 and len(filtered_end_times) > 0:
    for start, end in zip(filtered_start_times, filtered_end_times):
        burst_data = data[start:end]
    
        # Extract time and flux data for each burst
        burst_time = time_column[start:end]
        burst_flux = flux_data[start:end]
        burst_error = error_data[start:end]
        #normalized_flux = burst_flux / mean_flux    
        normalized_time = burst_time - burst_time[0]
    
        # Plot each detected burst
        plt.figure(figsize=(8, 6), dpi=300)
        #plt.errorbar(normalized_time, burst_flux, yerr=burst_error, fmt='-', color='red',  ecolor='gray', elinewidth=1, capsize=3)    
        plt.plot(normalized_time, burst_flux, color='red')
        plt.xlabel('Time [s]')
        plt.ylabel('Flux [counts/s]')
        plt.yscale('log')
        plt.title(f'Burst Light Curve from 4U 1728-34')
        #plt.axhline(y=1, color='gray', linestyle='--')
        #plt.legend()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.show()
        
        

        
        #print("normalised flux entries ", normalized_flux)
else:
    print("No bursts detected")
 