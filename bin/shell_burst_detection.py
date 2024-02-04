#!/bin/bash

# Directory containing the FITS files to process
input_directory="/path/to/input/fits/files"
# Existing directory where the burst-detected FITS files will be saved
output_directory="/export/data/oskarn/tests/bursts"
if [ ! -d "$output_directory" ]; then
    echo "Output directory $output_directory does not exist. Please create it before running this script."
    exit 1
fi

# Loop through all .fits files in the input directory
for file in "$input_directory"/*.fits; do
    echo "Processing file: $file"
    
    # Run your Python script here
    python your_script.py "$file" "$output_directory"
    
    echo "Finished processing $file"
done
