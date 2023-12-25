#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
code master's

"""

import subprocess
import sys
import os
import shutil
if len(sys.argv) < 2:
    print("No file name provided.")
    sys.exit(1)

selected_file = sys.argv[1]

selected_file = os.path.splitext(selected_file)[0]

command = ['saextrct']
# Start the saextrct process
process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

# Prepare the input data as a string, including all the necessary responses
input_data = '\n'.join([
   selected_file,
   '',  # Pressing enter for default value
   '',  # Pressing enter for default value
   'std1',
   '',  # Pressing enter for default value
   '',  # Pressing enter for default value
   '',  # Pressing enter for default value
   '1',  # This is where you specify the bin time in seconds
   '',  # Pressing enter for default value
   '',  # Pressing enter for default value
   '',
   '',
   '',
   '',
   '',
   '',
   '',
   '',

]) + '\n'

# Send the input data to the process and close stdin
output, errors = process.communicate(input_data)

# Check for errors and print the output
if errors:
   print(f"Errors: {errors}")
else:
   print(output)




##############################################################################################################
#############################################################################################################


# Get the current directory
current_directory = os.getcwd()

# Split the path into its components

path_components = current_directory.split(os.sep)

# Get the name of the 6th directory
sixth_directory = path_components[6] if len(path_components) > 5 else None

# Create the output file name
output_file_name = f"{sixth_directory}_std1.asc"


fdump_command = ['fdump', 'std1.lc', os.path.join(current_directory, output_file_name),'prhead=no', 'showrow=-', 'showunit=-', 'showcol=-']

process_fdump = subprocess.Popen(fdump_command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

# Prepare the input data as a string, including all the necessary responses
input_data2 = '\n'.join([
   '',
   '',  # Pressing enter for default value

]) + '\n'

output, errors = process_fdump.communicate(input_data2)

# Check for errors and print the output for Fdump
if errors:
   print(f"Errors: {errors}")
else:
   print(output)

current_location = os.path.join(current_directory, output_file_name)
new_location = "/export/data/oskarn/testy_skrypty"

# Move the file
shutil.move(current_location, new_location)
