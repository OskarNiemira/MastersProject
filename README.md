# Master's Dissertation Project on LMXBs

## Overview
This repository contains the scripts and tools developed for my Master's dissertation focusing on the analysis of Low-Mass X-ray Binaries (LMXBs). It includes a series of scripts designed to process raw data from RXTE observations, facilitating the extraction and analysis of light curves and burst features for subsequent machine learning analysis.

## Repository Structure
- **/bin/**: Contains shell scripts to prepare and process RXTE data.
  - `shell_file_search`: Filters and identifies PCA observations within the data.
  - `shell_burst_detection`: Automates running `saextrct` on specified directories to extract light curves.
- **/MastersProject/**: Contains Python scripts for further data manipulation and analysis.
  - `lightcurve_detection_updated_version`: Filters light curves and isolates burst data.
  - `normalise`: Normalizes observations for consistent analysis.
  - `burst_feature`: Extracts required parameters from FITS files and stores them in an Excel spreadsheet.
  - `ML for clusters`: Applies machine learning clustering techniques to categorize bursts based on     extracted features, aiding in the identification of underlying patterns or anomalies.
  - `burst_cropper`: This script processes the light curve files to crop out and separate burst data from non-burst data, focusing analysis on the events of interest.
  - `saextrct_automatic_run_2`: A Python automation script that interfaces with the `saextrct` tool from HEAsoft to streamline the extraction of light curves from multiple data files

## Getting Started

### Prerequisites
- HEAsoft
- Python 3
- Additional Python libraries: astropy, numpy, matplotlib, pandas, sklearn, os, sys, shutil, subprocess

  
