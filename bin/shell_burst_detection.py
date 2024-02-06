#!/bin/bash

# This script assumes your Python script is named "myscript.py"
SCRIPT_NAME="lightcurve_detection.py"

# This script assumes your target directory is provided as the first argument
TARGET_DIR="$1"

# Check if the directory exists
if [ ! -d "$TARGET_DIR" ]; then
 echo "Error: Directory '$TARGET_DIR' does not exist."
 exit 1
fi

# Loop through all files in the directory
for file in "$TARGET_DIR"/*; do
 # Skip directories and hidden files
 if [[ -d "$file" || "$file" == .*. ]]; then
   continue
 fi

 # Run the Python script with the current file as an argument
 python "$SCRIPT_NAME" "$file"

 # Add error handling (optional)
 if [ $? -ne 0 ]; then
   echo "Error: Python script failed on '$file'."
   exit 1
 fi
done

echo "Finished processing files in '$TARGET_DIR'."
