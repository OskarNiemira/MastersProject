import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

def normalize_and_plot_bursts(directory):
    plt.figure(figsize=(8, 6), dpi=300)
    
    for filename in os.listdir(directory):
        if filename.endswith('.fits'):
            file_path = os.path.join(directory, filename)
            
            with fits.open(file_path) as hdul:
                # Check if the expected HDU index exists
                if len(hdul) > 1:
                    mean_flux = hdul[0].header.get('MEANFLUX', 1)  # Use a default value of 1 if MEANFLUX is not present
                    data = hdul[1].data
                    
                    # Check if 'TIME' and 'RATE' columns exist
                    if 'TIME' in data.columns.names and 'RATE' in data.columns.names:
                        time_column = data['TIME']
                        flux_data = data['RATE']
                        
                        # Ensure that there's data to work with
                        if time_column.size > 0 and flux_data.size > 0:
                            # Normalize the flux data using MEANFLUX
                            normalized_flux = flux_data / mean_flux
                            plt.plot(time_column - time_column[0], normalized_flux, label=f'{filename[:-5]}')
                        else:
                            print(f"No data in TIME or RATE column for file {filename}")
                    else:
                        print(f"Expected columns not found in file {filename}")
                else:
                    print(f"Not enough HDUs in file {filename}")
    plt.xlabel('Time since burst start (s)')
    plt.ylabel('Normalized Flux')
    plt.title('Normalized Bursts from 4U 1636-536')
    plt.yscale('log')
    plt.axhline(y=1, color='gray', linestyle='--')
    plt.show()

# Replace "path_to_your_directory" with the actual directory containing your FITS files
normalize_and_plot_bursts(r"/Users/oskarniemira/Desktop/Masters Project/lokalne testy/data/data_4u1636/cropped_bursts")

def normalize_and_plot_bursts_separately(directory):
    for filename in os.listdir(directory):
        if filename.endswith('.fits'):
            file_path = os.path.join(directory, filename)
            
            with fits.open(file_path) as hdul:
                if len(hdul) > 1 and 'MEANFLUX' in hdul[0].header:
                    mean_flux = hdul[0].header['MEANFLUX']
                    data = hdul[1].data
                    
                    if 'TIME' in data.columns.names and 'RATE' in data.columns.names:
                        time_column = data['TIME']
                        flux_data = data['RATE']
                        
                        if time_column.size > 0 and flux_data.size > 0:
                            normalized_flux = flux_data / mean_flux
                            
                            plt.figure(figsize=(8, 6))
                            plt.plot(time_column - time_column[0], normalized_flux, label=f'{filename[:-5]}')
                            
                            plt.xlabel('Time since burst start (s)')
                            plt.ylabel('Normalized Flux')
                            plt.title(f'Normalized Burst: {filename[:-5]}')
                            plt.yscale('log')
                            plt.axhline(y=1, color='gray', linestyle='--')
                            plt.legend()
                            plt.show()
                        else:
                            print(f"No data in TIME or RATE column for file {filename}")
                    else:
                        print(f"Expected columns not found in file {filename}")
                else:
                    print(f"Not enough HDUs or MEANFLUX missing in file {filename}")

#normalize_and_plot_bursts_separately("/Users/oskarniemira/Desktop/Masters Project/lokalne testy/data/data_4u1728/cropped")
