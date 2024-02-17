import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

def normalize_and_plot_bursts(directory):
    plt.figure(figsize=(12, 6))
    
    for filename in os.listdir(directory):
        if filename.endswith('.fits'):  # Ensure you're working with FITS files
            file_path = os.path.join(directory, filename)
            
            with fits.open(file_path) as hdul:
                mean_flux = hdul[0].header['MEANFLUX']  # Extract MEANFLUX from the primary HDU header
                data = hdul[1].data  # Assuming the burst data is in the second HDU
                time_column = data['TIME']
                flux_data = data['RATE']
                
                # Normalize the flux data using MEANFLUX
                normalized_flux = flux_data / mean_flux
                
                # Plot the normalized burst
                plt.plot(time_column - time_column[0], normalized_flux, label=f'{filename[:-5]}')  # Adjust the label as needed

    plt.xlabel('Time since burst start (s)')
    plt.ylabel('Normalized Flux')
    plt.title('Normalized Bursts')
    #plt.legend()
    plt.show()

# Replace "path_to_your_directory" with the actual directory containing your FITS files
normalize_and_plot_bursts(r"C:\Users\oskar\OneDrive - University of Southampton\Desktop\physics\year 4\MastersProject\lokalne testy\outputs\fits files\test2")
