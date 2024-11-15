import re
import os
import numpy as np
import pandas as pd

sub_id = 10014
folder_path = f"E:/MindlessReading/Data/s{sub_id}"

# List to hold the DataFrames
file_paths = []

# Walk through the directory and subdirectories
for root, dirs, files in os.walk(folder_path):
    for filename in files:
        # Check if the file ends with 'sample.csv'
        if filename.endswith("Sample.csv"):
            file_path = os.path.join(root, filename)
            # Read the CSV file into a DataFrame
            file_paths.append(file_path)

# read in the "metadata" file and get page start and end time
for filename in os.listdir(folder_path):
    if filename.endswith("features_whole.csv"):
        file_path = os.path.join(folder_path, filename)
        # Read the CSV file into a DataFrame
        df_metadata = pd.read_csv(file_path)
        print(f"File loaded: {file_path}")
        break  # Stop after finding the first match

# extract the run num: regular expression pattern to find the digit after 'r'
pattern = r'_r(\d+)'
# loop through sample file for each run
for file_path in file_paths:
    # get the run number
    run_num = int(re.search(pattern, file_path).group(1))
    # read in csv file
    df_sample = pd.read_csv(file_path)
    # declare new columns
    df_sample['page_num'] = np.nan
    df_sample['run_num'] = np.nan
    df_sample['is_MW'] = 0
    df_sample['sample_id'] = 'None'

    # get the page start and end time from metadata
    df_page = df_metadata[df_metadata['run']==run_num]
    
    # loopp through each page
    for _, page_info in df_page.iterrows():
        # get page information
        page_num = page_info['page_num'] + 1
        page_start = page_info['page_start']*1000
        page_end = page_info['page_end']*1000

        # use 1 second time window to extract raw eye samples
        sample_count = 1
        sample_start = page_start
        sample_end = sample_start + 1000
        while (sample_end <= page_end):
            # define sample ID
            sample_id = f'{sub_id}_r{run_num}_p{page_num}_s{sample_count}'
            # create mask for the current sample
            mask = (df_sample['tSample'] >= sample_start) & (df_sample['tSample'] < sample_end)
            df_sample.loc[mask, 'page_num'] = page_num
            df_sample.loc[mask, 'run_num'] = run_num
            df_sample.loc[mask, 'sample_id'] = sample_id

            # update sample start and end time
            sample_start = sample_end + 1
            sample_end = sample_start + 1000
            sample_count += 1

        # log in the mind-wandering info
        win_start = page_info['win_start']
        win_end = page_info['win_end']
        if (win_start > 0) and (win_start > 0):
            win_mask = (df_sample['tSample'] >= win_start*1000) & (df_sample['tSample'] < win_end*1000)
            df_sample.loc[win_mask, 'is_MW'] = 1
        

    # truncate the sample table
    end_ind = df_sample.index[df_sample['tSample'] == sample_end][0]
    df_sample_trunc = df_sample.iloc[:end_ind]

    df_sample_trunc.to_csv('test.csv')

    
    break
