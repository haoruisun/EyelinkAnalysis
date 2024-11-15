# This file contains an example of blink interpolation
# Created by HS 10/31/24

import glob
import sys
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# Add the src directory to sys.path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(src_path)
# import modules from src
import utils

# load data
folder_path = os.path.dirname(__file__) + "/../res"
# dfBlink
# Search for files ending with 'Blink.csv' in the folder
file_path = glob.glob(os.path.join(folder_path, '*Blink.csv'))
# Check if any matching files were found
if file_path:
    # Load the first matched file into a DataFrame
    dfBlink = pd.read_csv(file_path[0])
    print("Loaded dfBlink file:", file_path[0])
else:
    print("No file ending with 'Blink.csv' found in the folder.")

# dfSamples
# Search for files ending with 'Blink.csv' in the folder
file_path = glob.glob(os.path.join(folder_path, '*Sample.csv'))
# Check if any matching files were found
if file_path:
    # Load the first matched file into a DataFrame
    dfSamples = pd.read_csv(file_path[0])
    print("Loaded dfSample file:", file_path[0])
else:
    print("No file ending with 'Sample.csv' found in the folder.")

# dfSaccade
# Search for files ending with 'Blink.csv' in the folder
file_path = glob.glob(os.path.join(folder_path, '*Saccade.csv'))
# Check if any matching files were found
if file_path:
    # Load the first matched file into a DataFrame
    dfSaccade = pd.read_csv(file_path[0])
    print("Loaded dfSample file:", file_path[0])
else:
    print("No file ending with 'Saccade.csv' found in the folder.")

# get a copy of original dfSamples
dfSamples_original = dfSamples.copy()

# call function to interpolate data during blinks
utils.interpolate_blink(dfSamples, dfBlink, dfSaccade)

# plot data
# Define column names and set up subplots
columns = ['LX', 'LY', 'LPupil', 'RX', 'RY', 'RPupil']
# get time array 
times = np.array(dfSamples['tSample'])/1000     # convert to seconds
times = times - times[0]

# Plot each column in a subplot
fig, axs = plt.subplots(3, 2, figsize=(12, 12))
for i, col in enumerate(columns):
    row, col_position = divmod(i, 2)
    axs[row, col_position].plot(times, dfSamples_original[col], label='Original')
    axs[row, col_position].plot(times, dfSamples[col], label='Interploated')
    axs[row, col_position].set_title(col)
    axs[row, col_position].set_xlabel("Time")
    axs[row, col_position].set_ylabel("Value")

plt.legend()
plt.tight_layout()
plt.show()

# for col in columns:
#     plt.figure()
#     plt.plot(times, dfSamples_original[col], label='Original')
#     plt.plot(times, dfSamples[col], label='Interploated')
#     plt.title(col)
#     plt.legend()
#     plt.show()
