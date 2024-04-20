#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 13:36:11 2024

@author: oskarniemira
"""

import numpy as np
from astropy.io import fits
import os
import pandas as pd
from numpy import trapz

directory = r"/Users/oskarniemira/Desktop/Masters Project/lokalne testy/data/He bursts/data_1735/cropped"

def calculate_burst_duration(start_time, end_time):
    """
    Calculate the duration of a burst.
    
    """
    return end_time - start_time

def normalised_peak_flux(flux, mean_flux):
    """
    Find the peak flux of the burst.
    
    """
    normalized_flux = flux_data / mean_flux
    
    return np.max(normalized_flux)

def calculate_rise_time(time, flux):
    """
    Calculate the rise time of the burst, from its start to the peak flux.
    
    - The rise time to peak flux.
    """
    peak_flux_index = np.argmax(flux)
    return time[peak_flux_index] - time[0]

def calculate_decay_time(time, flux):
    """
    Calculate the decay time of a burst from its peak flux to a threshold level.
    """
    peak_flux_index = np.argmax(flux)

    decay_time = time[-1] - time[peak_flux_index]   
    
    return decay_time

def calculate_auc(time, flux):
    """
    Calculate the area under the curve (AUC) for the burst using the trapezoidal rule.

    :param time: array of time values
    :param flux: array of normalized flux values
    :return: total area under the curve (AUC)
    """
    auc = trapz(flux, time)
    return auc

features = []

for filename in os.listdir(directory):
    if filename.endswith('.fits'):  # Ensure you're working with FITS files
        file_path = os.path.join(directory, filename)
        
        with fits.open(file_path) as hdul:
            mean_flux = hdul[0].header['MEANFLUX']  # Extract MEANFLUX from the primary HDU header
            data = hdul[1].data  # Assuming the burst data is in the second HDU
            time_column = data['TIME']
            flux_data = data['RATE']
            
            normalized_flux = flux_data / mean_flux
            
            burst_start_time = time_column[0]
            burst_end_time = time_column[-1]
            burst_duration = calculate_burst_duration(burst_start_time, burst_end_time)
            peak_flux = normalised_peak_flux(flux_data, mean_flux)
            rise_time = calculate_rise_time(time_column, normalized_flux)
            decay_time = calculate_decay_time(time_column, flux_data)
            auc = calculate_auc(time_column, normalized_flux)
            
            print("Burst Duration: ", burst_duration)
            print("Peak Flux: ", peak_flux)
            print("rise time: ", rise_time)
            print("decay time: ", decay_time)

            burst_features = [burst_duration, peak_flux, rise_time, decay_time, auc]
            features.append(burst_features)

# Convert the collected features into a pandas DataFrame
feature_names = ['Burst Duration', 'Peak Flux', 'Rise Time', 'Decay Time', 'AUC']
df = pd.DataFrame(features, columns=feature_names)
            
# Save the DataFrame to a CSV file
output_directory = r"/Users/oskarniemira/Desktop/Masters Project/lokalne testy/data/csv data/He bursts"
output_file_name = "data_4U 1735.csv"
output_file_path = os.path.join(output_directory, output_file_name)
df.to_csv(output_file_path, index=False)
            