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

base_file_name = os.path.splitext(os.path.basename(selected_file))[0]
print("Plik na ktorym pracujesz ", base_file_name)
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


output_file_name = f"{base_file_name}_std1.lc"
print("output file name nazwa: ", output_file_name)

if os.path.exists('std1.lc'):
    os.rename('std1.lc', output_file_name)
    print(f"File renamed to {output_file_name}")
    file_renamed = True
else:
    print("The file std1.lc does not exist and cannot be renamed.")
    file_renamed = False

##############################################################################################################
#############################################################################################################


# Get the current directory
current_directory = os.getcwd()

print("nazwa current directory",current_directory)

current_location = os.path.join(current_directory, output_file_name)
new_location = "/export/data/oskarn/testy_skrypty"
print("current location a potem new location", current_location)
print(new_location)
# Try to move the file and print an error message if it fails
if file_renamed and os.path.exists(output_file_name):
    try:
        shutil.move(current_location, new_location)
        print(f"File moved to {new_location}")
    except Exception as e:
        print(f"Could not move the file: {e}")
else:
    print("File was not renamed. No file to move.")
