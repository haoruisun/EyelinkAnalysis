# UVM Glass Brain
# May 2022
# George Spearing

# Main file to run analysis on reading experiment.

# Created 05/2022 by George Spearing 
# Updated 10/31/2023 by HS
# Updated 11/3/2023 by HS - add function documentations
#                         - create a new repo for Mindless Reading analyses
# Updated 11/10/23 by HS - add group extraction
# Updated 11/24/23 by HS - match nearby words (index off by 1) to clicked words
#                        - use page end time for valid onset but unfound offset
# Updated 01/19/24 by HS - add another two types of analyses of eye features
#                        - pack repeated code into functions
# Updated 4/9/24 by HS - estimate MR on/offset by finding the maximum subarray
#                        see convert_click_to_time func for more details
# Updated 1/29/25 by HS - test and debug new pipeline with default and same-dur win_type
#                         enable feature extraction from sliding windows 
# Packages
import os
import re
import glob
import copy
import pickle
import random
import warnings
import numpy as np
import pandas as pd   
from tqdm import tqdm
from collections import deque              
from . import parse_EyeLink_asc as eyelink
from .utils import interpolate_blink
from .match_word import match_clicks2words
from .calculate_eye_features import calculate_all_features
from .plot_reading import plot_reading_results

# Functions
def extract_subject_features(sub_folder, win_type, is_plot=False):
    """
    Extracts and processes eye-tracking features for a given subject.

    This function serves as a wrapper to:
    1. Load precomputed page objects or generate them if necessary.
    2. Compute eye-tracking features using either:
       - A sliding window approach (`slide` mode)
       - A fixed window per page (`default` or `same-dur` mode)
    3. Optionally plot fixation results.
    4. Save extracted features into CSV files.

    Sliding Window vs. Fixed Window Approach
    The sliding window method **does not create new page objects** for each window shift.  
    Instead, it reuses the same page object, making it **more memory-efficient** compared to the fixed window approach.

    Parameters
    ----------
    sub_folder : str
        Path to the subject folder containing data.
    win_type : str
        Type of windowing method to use ('slide' for sliding window, otherwise fixed windowing).
    is_plot : bool, optional
        Whether to generate and save fixation plots. Default is False.

    Raises
    ------
    Exception
        If no PsychoPy file matches the eye-tracking file (i.e., run numbers do not match).

    Returns
    -------
    None
        Processed eye features (L & R) are saved as CSV files.
    """
    # extract subject ID
    sub_id = re.findall(r's\d+', sub_folder)[0]
    # get the last five digit ID
    sub_id = int(sub_id[-5:])
    # print beginning sentence
    print(f'\nBegin Extracting Eye Features for Subject {sub_id}...')
    print('====================================================================')

    # load pages
    overwrite_page = True
    all_pages = load_pages(sub_folder, overwrite_page)
    
    # for sliding window
    if win_type == 'slide':
        print('Computing eye features using a sliding window approach...')
        # Sliding window parameters
        slidewindow_len = 5
        time_offset = 5
        page_time_offset = time_offset + slidewindow_len / 2
        step = 0.25
        win_type = f'slide_wlen{slidewindow_len}'
        # lists to store results
        all_res_left, all_res_right = [], []

        for page in tqdm(all_pages, desc="Calculating features on each page"):
            if page.mw_reported:
                # Process MW onset
                mid_point = page.mw_onset
                process_sliding_window(page, mid_point, 'MW_onset', slidewindow_len, time_offset, step, all_res_left, all_res_right)

                # Process self-report event (offset)
                mid_point = page.time_end - slidewindow_len / 2
                process_sliding_window(page, mid_point, 'self_report', slidewindow_len, time_offset, step, all_res_left, all_res_right)

            else:
                # Generate random mid_point for control case
                mid_point = random.uniform(page.time_start + page_time_offset, page.time_end - page_time_offset)
                process_sliding_window(page, mid_point, 'control', slidewindow_len, time_offset, step, all_res_left, all_res_right)

         # Convert results to DataFrames
        df_L, df_R = pd.DataFrame(all_res_left), pd.DataFrame(all_res_right)

        # Normalize pupil size across trials to reduce individual differences
        for df in [df_L, df_R]:
            if not df.empty:
                df['norm_pupil'] = df['mean_pupil'] / df['mean_pupil'].mean()
    
    else:
        print('Computing window start and end times for each page...')
        all_pages = compute_window_time(all_pages, win_type)

        print('Extracting and calculating eye features...')
        df_L, df_R = compute_features(all_pages)

        if is_plot:
            print('Ploting fixations...')
            plot_reading_results(all_pages, sub_folder, win_type)

    # Save concatenated results
    if not df_L.empty:
        df_L.to_csv(f'{sub_folder}/s{sub_id}_L_features_{win_type}.csv')
        print('Saved Data for L Eye.')

    if not df_R.empty:
        df_R.to_csv(f'{sub_folder}/s{sub_id}_R_features_{win_type}.csv')
        print('Saved Data for R Eye.')
    
    # print finish sentence
    print('================================= Log =================================')
    print(f'Subject: {sub_id} has been DONE!\n')


def load_pages(sub_folder, overwrite_page):
    """
    Loads or generates page objects for a given subject.

    This function checks whether precomputed page objects exist in the subject's folder.
    If `overwrite_page` is True or the folder does not exist, it regenerates the page 
    objects from eye-tracking and PsychoPy log files. Otherwise, it loads precomputed 
    pages from pickle files.

    Parameters
    ----------
    sub_folder : str
        The path to the subject folder containing eye-tracking and log files.
    overwrite_page : bool
        Whether to overwrite existing page objects and regenerate them.

    Returns
    -------
    list
        A list of page objects, either loaded from pickle files or newly created.
    """

    # Define the directory where page objects are stored
    dir_page = os.path.join(sub_folder, "page")

    # Ensure the page directory exists, and set overwrite flag if newly created
    if not os.path.exists(dir_page):
        print("Generating page objects...")
        os.makedirs(dir_page)
        overwrite_page = True

    # List to store all loaded/generated pages
    all_pages = []

    # If overwrite or page folder doesn't exist, read in eye-tracking and psychopy files to 
    # generate page objects and save them to the folder
    if overwrite_page:
        # Define directories for eye-tracking and behavioral log files
        dir_eye, dir_log = os.path.join(sub_folder, "eye"), os.path.join(sub_folder, "log")
        # Get sorted lists of eye-tracking and PsychoPy files
        eye_files, beh_files = np.sort(os.listdir(dir_eye)), np.sort(os.listdir(dir_log))   

         # Loop through eye-tracking files and find matching PsychoPy files
        for eye_file in eye_files:
            eye_index = extract_run_index(eye_file, ".asc")
            if eye_index < 0:  # Skip invalid indices
                continue
        
            # boolean variable to make sure every eye file matches to a psychopy file
            is_match = False
            for beh_file in beh_files:
                # extract run index from PsychoPy .csv file
                beh_index = extract_run_index(beh_file, '.csv')
                # for the same run/session number
                if eye_index == beh_index:
                    is_match = True
                    print(f'\n============================ Run{eye_index} Log ==============================')
                    print(f'Eye-Tracking File Name: {eye_file}')
                    print(f'PsychoPy File Name: {beh_file}')
                    # Define the name of the file that stores page objects for this run
                    page_file_path = os.path.join(dir_page, f'r{eye_index}_pages')
                    # Define file paths for eye-tracking and PsychoPy log files
                    eye_file_path = os.path.join(dir_eye, eye_file)
                    psy_file_path = os.path.join(dir_log, beh_file)

                    pages = process_data2pages(eye_file_path, psy_file_path)
                    # Save page objects to a pickle file
                    print("Saving page objects to file...")
                    with open(page_file_path, 'wb') as f:
                        pickle.dump(pages, f)
                    # Append pages to the main list
                    all_pages.extend(pages)
                    break  # Skip remaining PsychoPy files once a match is found

            # raise an exception if no PsychoPy file matches eye-tracking file
            if not is_match:
                raise Exception(f'Eye file: {eye_file} has no matched PsychoPy file!')
    
    # directly read page objects from the file
    else:
        # Load precomputed page objects from pickle files
        print("Loading precomputed page objects from files...")
        # Get all matching pickle files
        for file in os.listdir(dir_page):
            file_path = os.path.join(dir_page, file)
            with open(file_path, "rb") as f:
                pages = pickle.load(f)  # Assuming each file contains a list of pages
            all_pages += pages
    
    return all_pages


def process_data2pages(eye_file_path, psy_file_path):
    '''
    _summary_

    Args:
        eye_file_path (_type_): _description_
        psy_file_path (_type_): _description_

    Returns:
        _type_: _description_
    '''    
    # extract run index from eye file
    run_number = extract_run_index(eye_file_path, '.asc')
    # get the data in pandas dataframe
    dfTrial,dfMsg,dfFix,dfSacc,dfBlink,dfSamples = eyelink.load_data(eye_file_path)
    # interpolate samples with blink and saccade info
    dfSamples = interpolate_blink(dfSamples, dfBlink, dfSacc)
    # generate page objects 
    print('Generate Page objects for each reading page...')
    pages = eyelink.generate_pages(dfMsg) # returns a list of page objects

    # match clicks to words
    matched_words = match_clicks2words(psy_file_path)

    print('Match fixations to words and compute MW onset and offset...')
    # loop thru each page
    for page in pages:
        # store features only for the 1st pass pages
        if(page.page_view != '1st Pass'):
            continue
        
        # assign run number
        page.run_number = run_number
        # assign eye feature dataframe to fields
        page.assign_data(dfFix, dfBlink, dfSacc, dfSamples)
        # match fixtions to words
        page.match_fix2words()

        # for any reported page
        if type(matched_words['first_word'].iloc[page.page_number]) == str:
            # call function to estimate the MW onset and offset
            page.find_MW_time(matched_words)

    return pages


def compute_features(pages):
    '''
    _summary_

    Args:
        pages (_type_): _description_
        is_plot (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    '''
    # lists to store results
    all_res_left, all_res_right = [], []
    # loop thru each page
    for page in pages:
        res_left, res_right = None, None
        # calculate eye features with defined time window
        res_left, res_right = calculate_all_features(page)
        
        # save current page results
        if res_left is not None:
            # fill in other information (i.e. page, time info etc)
            fill_dict(res_left, page)
            # append the res dict to list
            all_res_left.append(res_left)
        
        if res_right is not None:
            fill_dict(res_right, page)
            all_res_right.append(res_right)

    # save results
    df_L, df_R = pd.DataFrame(all_res_left), pd.DataFrame(all_res_right)

    # modify dataframe
    for df in [df_L, df_R]:
        if not df.empty:
            # normalize some features to avoid individual differences
            df['norm_pupil'] = df['mean_pupil'] / df['mean_pupil'].mean()
         
    return df_L, df_R



def compute_window_time(pages, win_type):
    '''
    Assigns window start and end times to pages based on the specified window type.

    Parameters:
        pages (list): List of page objects.
        win_type (str): Window type, either 'default' or 'same-dur'.

    Returns:
        list: Updated list of page objects with window time information.
    '''
    # default type
    # normal reading sample: page start -> page end
    # mindless reading (MW) sample: mw onset -> mw offset
    if win_type == 'default':
        # loop thru all pages and update win start/end info
        for page in pages:
            # MW sample
            if page.mw_reported:
                page.win_start, page.win_end = page.mw_onset, page.mw_offset
            # normal sample
            else:
                page.win_start, page.win_end = page.time_start, page.time_end
        return pages

    # same-dur type
    # match MR and NR window length
    elif win_type == 'same-dur':
        # declare empty list to store page objects
        pages_nr = []   # list of normal reading page
        pages_mw = []   # list of mw page

        for page in pages:
            if page.mw_dur >= 2:
                page.win_start = page.mw_onset
                page.win_end = page.mw_offset
                pages_mw.append(page)
            elif not page.mw_reported:
                pages_nr.append(page)
        
        if len(pages_mw) == 0:
            raise ValueError('No mind-wandering instances are found!\nThrow out this subject from analysis')

        # Sort pages for pairing
        # sort MW pages based on descending order of MW duration
        # sort MW pages again based on ascending order of time diff between page start and onset
        pages_mw.sort(key=lambda p: (p.mw_onset - p.time_start, -p.mw_dur))
        # sort normal pages based on descending order of page duration
        pages_nr.sort(key=lambda p: -p.page_dur)

        # if there are no normal reading pages, define normal time window (NR) from reading periods # before the first click word, aka, MW onset
        # those pages are saved in the list "pages_nr_mw"
        if len(pages_nr) == 0:
            pages_nr_mw = pages_mw
        # if there are more mind wandering pages than normal reading pages
        elif len(pages_mw) > len(pages_nr):
            # extract normal reading time window from extra mind wandering pages
            pages_nr_mw = pages_mw[len(pages_nr):]
            pages_mw = pages_mw[:len(pages_nr)]
        # if there are less mind wandering pages than normal reading pages
        # use each MW page to define window length on noraml pages    
        else:
            pages_nr_mw = []

        # loop thru each normal reading page and define time window based on mindless reading 
        # samples from all 5 runs
        for index, page_nr in enumerate(pages_nr):
            page_mw = pages_mw[index % len(pages_mw)]
             # calculate offsets and duration for the mind-wandering window
            mw_onset = page_mw.mw_onset - page_mw.time_start
            mw_offset = page_mw.mw_offset - page_mw.time_start
            mw_dur = page_mw.mw_offset - page_mw.mw_onset

            if page_nr.time_start + mw_offset > page_nr.time_end:
                win_start = random.uniform(page_nr.time_start, page_nr.time_end-mw_dur)
            else:
                win_start = page_nr.time_start + mw_onset
            win_end = win_start + mw_dur
            page_nr.win_start, page_nr.win_end = win_start, win_end

        for page_nr_mw in pages_nr_mw:
            # calculate MW duration and time between page start and MW onset
            mw_dur = page_nr_mw.mw_offset - page_nr_mw.mw_onset
            nr_dur = page_nr_mw.mw_onset - page_nr_mw.time_start

            # for page with normal reading time shorter than MW, skip
            if mw_dur > nr_dur:
                continue
            
            # copy the current page object and change all MW settings to False/np.nan
            page_nr = copy.deepcopy(page_nr_mw)
            page_nr.mw_reported = False
            page_nr.mw_valid = False

            # pick up a random time for normal window start time
            win_start = random.uniform(page_nr.time_start, page_nr.mw_onset-mw_dur)
            win_end = win_start + mw_dur

            # update win start and end info
            page_nr.win_start, page_nr.win_end = win_start, win_end
            page_nr.mw_onset, page_nr.mw_offset = np.nan, np.nan
            # save the normal reading page to return list
            pages_nr.append(page_nr)

        # update the return variable by adding normaing reading page, mindless reading page, normal reading duration from mindless reading page, mind-wandering duration from mindless reading page. 
        return pages_nr + pages_mw + pages_nr_mw


def fill_dict(res, page, verbose=True):
    '''
    Populates a dictionary (`res`) with metadata and time-related information 
    from a given `page` object. Optionally includes mind-wandering (MW) 
    details if `verbose` is set to True.

    Parameters
    ----------
    res : dict
        The dictionary to be populated with metadata and time-related values.
    page : object
        An object containing attributes related to the reading session.
    verbose : bool, optional
        If True, includes MW-related information in the dictionary. 
        Defaults to True.

    Returns
    -------
    res: dict
        The updated dictionary containing page metadata.

    ''' 
    reading_convert = {'history_of_film':'History of Film',
                       'pluto':'Pluto',
                       'serena_williams':'Serena Williams',
                       'the_voynich_manuscript':'The Voynich Manuscript',
                       'prisoners_dilemma':'Prisoners Dilemma'}
    
    res['reading'] = reading_convert[page.reading]
    res['run'] = page.run_number
    res['page'] = page.page_number
    res['page_start'] = page.time_start
    res['page_end'] = page.time_end
    res['win_start'] = page.win_start
    res['win_end'] = page.win_end
    res['win_dur'] = page.win_dur
    res['task_start'] = page.task_start

    # store MW related information for verbose mode
    if verbose:
        res['is_MWreported'] = page.mw_reported
        res['is_MWvalid'] = page.mw_valid
        res['MW_start'] = page.mw_onset
        res['MW_end'] = page.mw_offset

    return res

    
def extract_run_index(file_name, file_type, re_pattern=r'r[0-9]'):
    '''
    This funciton uses regular experssion to extract run/session index number

    Parameters
    ----------
    file_name : string
        DESCRIPTION. file name
    file_type : string
        DESCRIPTION. file type 
            .asc for eye tracking data
            .csv for PsychoPy file
    re_pattern : string
        DESCRIPTION. The pattern for run/session index number
        Default: *_r[0-9]_*

    Returns
    -------
    run_index : int
        DESCRIPTION. the run/session index number extracted from input file

    '''
    # check whether the file name ends with specified file type
    if (file_name.endswith(file_type)):
        # use the regular pattern to extract the run/session index number
        run_index = re.search(re_pattern, file_name, re.IGNORECASE)
        run_index = int(run_index[0][-1])
    
    # if not meets the file type, return -1 for the run_index
    else:
        run_index = -1
    
    return run_index



def process_sliding_window(page, mid_point, label, slidewindow_len, time_offset, step, all_res_left, all_res_right):
    """
    Processes eye features within a sliding window around a given mid_point.

    Parameters
    ----------
    page : Page object
        The page object containing eye-tracking data.
    mid_point : float
        The central time point around which the sliding window is applied.
    label : str
        Label indicating the type of event (e.g., 'MW_onset', 'self_report', 'control').
    slidewindow_len : float
        Length of the sliding window.
    time_offset : float
        The maximum allowed offset for backward and forward sliding.
    step : float
        Step size for the sliding window.
    all_res_left : list
        List to store left eye results.
    all_res_right : list
        List to store right eye results.
    """

    # handle the case where no time points were assigned to MW onset/offset
    if np.isnan(mid_point):
        print("Warning: Cannot convert float NaN to integer. Time point was not found.")
        return

    total_steps = time_offset / step
    page_copy = copy.deepcopy(page)

    # Compute backward and forward steps
    b_time = mid_point - page.time_start
    backward_steps = min(total_steps, int((b_time - slidewindow_len) / step))

    f_time = page.time_end - mid_point
    forward_steps = min(total_steps, int((f_time - slidewindow_len) / step))

    left = mid_point - backward_steps * step - slidewindow_len / 2
    right = mid_point + forward_steps * step - slidewindow_len / 2

    # Iterate over sliding window steps
    for win_start in np.arange(left, right + step, step):
        page_copy.win_start = win_start
        page_copy.win_end = win_start + slidewindow_len
        relative_time = win_start - (mid_point - slidewindow_len / 2)

        res_left, res_right = calculate_all_features(page_copy)

        # Store left eye results
        if res_left is not None:
            fill_dict(res_left, page_copy, verbose=False)
            res_left['relative_time'] = relative_time
            res_left['label'] = label
            all_res_left.append(res_left)

        # Store right eye results
        if res_right is not None:
            fill_dict(res_right, page_copy, verbose=False)
            res_right['relative_time'] = relative_time
            res_right['label'] = label
            all_res_right.append(res_right)

                

def extract_pupil_trace(page, duration=6):
    '''
    This function takes page object and extract pupil samples for a total
    duration of 6 seconds for three different types:
        onset: MW onset ---------6s--------->
        offset: <---------6s--------- MW offset
        control (ctr): <---------6s--------- MW onset
    Pupil samples are nomalized by subtracting average pupil size of piror MW
    period (i.e. page start time -------> MW onset)

    Parameters
    ----------
    page : Tobject
        DESCRIPTION. See page.py for more details. 
    duration : int, optional
        DESCRIPTION. The duration of pupil trace window. 
            The default is 6.
            Ref: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0226792

    Returns
    -------
    onset_array : numpy float array, len = duration x sampling fs
        DESCRIPTION.
    offset_array : numpy float array, len = duration x sampling fs
        DESCRIPTION.
    ctr_array : numpy float array, len = duration x sampling fs
        DESCRIPTION.

    '''
    # define the sampling frequency for sample data
    # original 1000 Hz, but resampled to 100 Hz
    fs = 100
    # define the duration in sample count
    epoch_dur = fs * duration
    
    # get the sample dataframe
    df = page.pupils
    pupils = np.array(df['pupil'])
    time = df['tSample']
    
    # find the index for onset pupil trace 
    ind = np.argmin(np.abs(time-page.mw_onset*1000))
    # compute the baseline
    baseline = np.mean(pupils[0:ind])
    # extract the pupil trace into an array
    onset_array = np.array(pupils[ind:ind+epoch_dur]-baseline)
    
    # extract pupil trace before the onset as the control group
    ctr_array = np.array(pupils[ind-epoch_dur:ind]-baseline)
    
    # find the index for offset pupil trace 
    ind = np.argmin(np.abs(time-page.mw_offset*1000))
    # extract the pupil trace into an array
    offset_array = np.array(pupils[ind-epoch_dur:ind]-baseline)
            
    # return pupil epochs
    return onset_array, offset_array, ctr_array


