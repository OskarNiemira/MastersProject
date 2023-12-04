# -*- coding: utf-8 -*-
"""
code master's
"""

#!/usr/bin/env python3
import subprocess

command = ['saextrct']
# Start the saextrct process
process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

# Prepare the input data as a string, including all the necessary responses
input_data = '\n'.join([
   'FS46_5366055-5366790',
   '',  # Pressing enter for default value
   '',  # Pressing enter for default value
   'std6',
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




fdump_command = ['fdump', 'std6.lc+1', 'std6.asc','prhead=no', 'showrow=-', 'showunit=-', 'showcol=-']

process_fdump = subprocess.Popen(fdump_command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

# Prepare the input data as a string, including all the necessary responses
input_data2 = '\n'.join([
   '',
   '',  # Pressing enter for default value
   '',  # Pressing enter for default value
   'std6',
]) + '\n'

output, errors = process_fdump.communicate(input_data2)

# Check for errors and print the output for Fdump
if errors:
   print(f"Errors: {errors}")
else:
   print(output)
