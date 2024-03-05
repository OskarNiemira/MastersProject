# -*- coding: utf-8 -*-
"""
Code meant to crop bursts from a file and save them all separately
It also does further filtring of bursts, as some of the bursts are no actual bursts

@author: oskar
"""

import astropy.io.fits as fits
from astropy.table import Table
import os
import numpy as np
import sys
import matplotlib.pyplot as plt

# Define the file path and output directory directly
input_directory = r"/Users/oskarniemira/Desktop/Masters Project/lokalne testy/data"  
output_directory = r"/Users/oskarniemira/Desktop/Masters Project/lokalne testy/outputs/fits files/test4"
print(f"Found {len(os.listdir(input_directory))} files in the directory.")
for file_name in os.listdir(input_directory):
    print(f"Processing file: {file_name}")
    if file_name.endswith(".lc"):
        file_path = os.path.join(input_directory, file_name)

        try:
            # Load the RXTE data from the FITS file
            with fits.open(file_path) as hdul:
                data = hdul['RATE'].data
                time_column = data['TIME']
                flux_data = data['RATE']
            print("Data loaded successfully from {}".format(file_path))
        
            # Plot the original light curve
            plt.figure(figsize=(10, 4))
            plt.plot(time_column, flux_data, label='Original Data')
            plt.xlabel('Time (MJD)')
            plt.ylabel('Flux')
            plt.title('Original Light Curve')
            plt.yscale('log')
            plt.legend()
            plt.show()
        except Exception as e:
            print("Failed to load data: {}".format(e))
            sys.exit(1)
        
        # Calculate mean and standard deviation excluding zero values (if zeros indicate no observation)
        valid_flux_data = flux_data[flux_data > 750]
        mean_flux = np.mean(valid_flux_data)
        std_flux = np.std(valid_flux_data)
        
        print("Mean: ", mean_flux)
        print("Std: ", std_flux)
        # Define the burst threshold as 4 times the standard deviation above the mean
        burst_threshold = mean_flux + 4.5 * std_flux
        print("Burst detection threshold:", burst_threshold)
        absolute_threshold = 2500
        
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
        print("bursts start: ", burst_indices_start)
        print("bursts end: ", burst_indices_end)
        
        def find_initial_growing_phase(flux, start_times, std_flux, lookback=20):
            """
            Adjusts burst start times to include the initial growing phase.
            
            :param flux: The flux data array.
            :param start_times: The initially detected start times of bursts.
            :param lookback: The maximum number of points to look back for the growing phase.
            :return: Adjusted start times including the initial growing phase.
            """
            adjusted_start_times = []
            threshold_point = std_flux * 2
            for start in start_times:
                adjusted_start_found = False
                for i in range(start, max(start - lookback, 1), -1):  # Ensure i-1 is valid by starting from max(start-lookback, 1)
                    flux_increase = flux[i] - flux[i - 1]
                    if flux_increase > threshold_point:
                        adjusted_start_times.append(i - 2)  # Append the point before the large increase
                        adjusted_start_found = True
                        break
                if not adjusted_start_found:
                    # If no adjusted start found within the lookback period, default to the initial start time
                    adjusted_start_times.append(start)
            return adjusted_start_times
        
        # Use the function to adjust start times
        adjusted_start_times = find_initial_growing_phase(flux_data, burst_indices_start, std_flux)
        
        def adjust_finish_times(flux, end_times, mean_flux, std_flux, extension=20):
            """
            Adjusts burst end times based on when the flux drops to mean + 2 * std.
            
            :param flux: The flux data array.
            :param end_times: The initially detected end times of bursts.
            :param mean_flux: Mean flux calculated from non-burst periods.
            :param std_flux: Standard deviation of flux from non-burst periods.
            :param extension: The number of points to look forward for stabilization.
            :return: Adjusted end times to include the gradual drop-off phase.
            """
            adjusted_end_times = []
            end_threshold = mean_flux 
        
            for end in end_times:
                adjusted_end = end #+ std_flux
                for i in range(end, min(end+extension, len(flux))):
                    # Check if the flux has dropped below the adjusted threshold
                    if flux[i] <= end_threshold:
                        adjusted_end = i
                        # Verify if the flux remains below the threshold for a grace period
                        grace_period = 10  # A small number of points to ensure stability
                        if i + grace_period < len(flux) and all(flux[j] <= end_threshold for j in range(i, i+grace_period)):
                            break
                    else:
                        # If the flux goes back above the threshold, reset the adjusted end
                        adjusted_end = i
                adjusted_end_times.append(adjusted_end)
            
            return adjusted_end_times
        
        adjusted_end_times = adjust_finish_times(flux_data, burst_indices_end, mean_flux, std_flux)
        
        print("adjusted bursts start: ", adjusted_start_times)
        print("adjusted bursts end: ", adjusted_end_times)
        
        def filter_bursts_by_flux_change(time_column, flux_data, start_indices, end_indices, window_size=600, threshold=0.1):
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
        
        for start, end in zip(filtered_start_times, filtered_end_times):
            burst_data = data[start:end]
        
            # Extract time and flux data for each burst
            burst_time = time_column[start:end]
            burst_flux = flux_data[start:end]
        
            # Plot each detected burst
            plt.figure(figsize=(10, 4))
            plt.plot(burst_time, burst_flux, label=f'Detected Burst from {start} to {end}', color='red')
            plt.xlabel('Time (MJD)')
            plt.ylabel('Flux')
            plt.title(f'Detected Burst Light Curve from index {start} to {end}')
            plt.yscale('log')
            plt.legend()
            plt.show()
            
        def crop_and_save_as_fits(data, time_column, flux_data, start_times, end_times, mean_flux, file_path, output_directory):
            """
            Crops the bursts from the data and saves them as individual FITS files.
        
            """
            base_file_name = os.path.basename(file_path)
            base_file_name_without_ext = os.path.splitext(base_file_name)[0]
            
            mean_flux_array = np.array([mean_flux])

            for i, (start, end) in enumerate(zip(start_times, end_times)):
                # Crop the burst data
                burst_time = time_column[start:end+1]
                burst_flux = flux_data[start:end+1]
                
                # Create a new table with the cropped data
                burst_table = Table([burst_time, burst_flux], names=('TIME', 'RATE'))
                burst_hdu = fits.BinTableHDU(burst_table)
                
                # Create a primary HDU for the burst data
                primary_hdu = fits.PrimaryHDU()
                primary_hdu.header['MEANFLUX'] = (mean_flux, 'Mean flux of the continuum')
                
                # Combine HDUs into an HDUList
                hdulist = fits.HDUList([primary_hdu, burst_hdu])
                
                # Define the file name for the cropped burst
                output_file_name = f"{base_file_name_without_ext}_burst_{i}.fits"
                output_file_path = os.path.join(output_directory, output_file_name)
                
                # Write the burst data to a new FITS file
                hdulist.writeto(output_file_path, overwrite=True)
                print(f"Cropped burst data saved to {output_file_path}")
        

        crop_and_save_as_fits(data, time_column, flux_data, adjusted_start_times, adjusted_end_times,mean_flux, file_path, output_directory)
        


