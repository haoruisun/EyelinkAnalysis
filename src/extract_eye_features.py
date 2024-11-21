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
# Packages
import os
import re
import glob
import copy
import pickle
import random
import numpy as np
import pandas as pd   
from collections import deque              
from . import parse_EyeLink_asc as eyelink
from .utils import interpolate_blink
from .match_word import match_clicks2words
from .calculate_eye_features import calculate_all_features

# Functions


def extract_group_features(data_path='../../../Data/', win_type='default'):
    '''
    This function finds all individual subject folder and call function to them
    to extract eye features. 

    Parameters
    ----------
    root_path : srtring, optional
        DESCRIPTION. The relative path of data collection folder
        The default is '../../../Data/'.
    mono_dataset : string list, optional
        DESCRIPTION. Contains information about the mono dataset
        For example, if s10001 was recorded using mono mode, mono_dataset=[10001]
        The default is [].

    Returns
    -------
    None.

    '''
    # get all subject folders in the root path
    subject_folders = glob.glob(f'{data_path}s[0-9]*')
    # loop through individual subject
    for subject_folder in subject_folders:
        # extract subject ID
        sub_id = re.findall(r's\d+', subject_folder)[0]
        # get the last five digit ID
        sub_id = int(sub_id[-5:])

        # run function on that subject
        extract_subject_features(sub_id, win_type=win_type)
            

def extract_subject_features(sub_id, win_type='default'):
    '''
    This function serves as a wrapper that calls other function to parse and 
    analyze eye features, 

    Parameters
    ----------
    sub_path : string
        DESCRIPTION. subject folder path
    mode : string, optional
        DESCRIPTION. eye-tracker recording mode: binocular or monocular 
            The default is 'bino'.

    Raises
    ------
    Exception
        DESCRIPTION. If no PsychoPy file matches eye-tracking file, i.e. run 
        numbers are not matched
                                                                
    Returns
    -------
    None. Subject eye features (L&R or mono) will be saved as csv file(s)

    '''
    # print beginning sentence
    print(f'\nBegin to Extract Eye Features for Subject {sub_id}...')

    # Define directories
    sub_path = f'../../../Data/s{sub_id}'
    dir_eye, dir_log = f'{sub_path}/eye/', f'{sub_path}/log/'
    eye_files, beh_files = np.sort(os.listdir(dir_eye)), np.sort(os.listdir(dir_log))

    # make the folder to store page objects
    dir_page = f'{sub_path}/page/'
    os.makedirs(dir_page, exist_ok=True)
    overwrite_page = False
    
    # Results storage
    res_L, res_R = [], []

    # loop thru
    for eye_file in eye_files:
        # extract run index from eye file
        eye_index = extract_run_index(eye_file, '.asc')
        if eye_index < 0:
            continue
        else:
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

                    # define the name of the file that stores page objects for this run
                    page_file_path = dir_page + f'r{eye_index}_pages'

                    if not os.path.exists(page_file_path) or overwrite_page:
                        # call function to extract eye features for a single run
                        eye_file_path = dir_eye + eye_file
                        psy_file_path = dir_log + beh_file

                        pages = process_data2pages(eye_file_path, psy_file_path)

                        # save page objects into the file
                        print('Save page objects into the file...')
                        with open(page_file_path, 'wb') as file:
                            pickle.dump(pages, file)

                    # read data directly from the page object file
                    else:
                        print('Directly read page objects from the file...')
                        with open(page_file_path, 'rb') as file:
                            pages = pickle.load(file)

                    print('Extract and calculate eye features from each page... ')
                    df_L_, df_R_ = extract_run_features(pages, win_type)

                    if not df_L_.empty:
                        res_L.append(df_L_)
                    if not df_R_.empty:
                        res_R.append(df_R_)

            # raise an exception if no PsychoPy file matches eye-tracking file
            if not is_match:
                raise Exception(f'Eye file: {eye_file} has no matched PsychoPy file!')
    
    # Save concatenated results
    if res_L:
        pd.concat(res_L, axis=0, ignore_index=True).to_csv(f'{sub_path}/s{sub_id}_L_features_{win_type}.csv')
        print('Saved Data for L Eye.')

    if res_R:
        pd.concat(res_R, axis=0, ignore_index=True).to_csv(f'{sub_path}/s{sub_id}_R_features_{win_type}.csv')
        print('Saved Data for R Eye.')

    
    #df_pupil.to_csv(dir_prefix + f'/s{subject}_PupilTrace.csv')
    
    # print finish sentence
    print('================================= Log =================================')
    print(f'Subject: {sub_id} has been DONE!\n')


def process_data2pages(eye_file_path, psy_file_path):
    '''
    _summary_

    Args:
        eye_file_path (_type_): _description_
        psy_file_path (_type_): _description_

    Returns:
        _type_: _description_
    '''    
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
        
        # assign eye feature dataframe to fields
        page.assign_data(dfFix, dfBlink, dfSacc, dfSamples)

        # match fixtions to words
        page.match_fix2words()

        # for any reported page
        if type(matched_words['first_word'].iloc[page.page_number]) == str:
            # call function to estimate the MW onset and offset
            page.find_MW_time(matched_words)

    return pages


def extract_run_features(pages, win_type, dataPlot=True):
    '''
    main body to control parsing and feature extraction
    The variables here come from the argParse located at the end of this file. 

    input:  fileDir - directory where the .asc files live
            fileName - filename for the .asc file
            dataPlot - type of plot to display once features are extracted
    output: None, shows plots if user selected
    '''
    
    if win_type == 'same-dur':
        list_normal = []
        list_mw = []
        # loop thru each page
        for page in pages:
            # for any reported page
            if np.isnan(page.mw_onset):
                # categorize page based on MW report
                list_normal.append(page)
            else:
                list_mw.append(page)

        # sort normal page based on page duration
        q_normal = deque(sorted(list_normal, key=lambda page: page.page_dur, reverse=True))
        # sort mand-wandering page based on mw duration
        q_mw = deque(sorted(list_mw, key=lambda page: page.mw_offset - page.mw_onset, reverse=True))

        pages, win_time = compute_window_time(pages, q_normal, q_mw)
            
    # lists to store results
    all_res_left, all_res_right = [], []
    # loop thru each page
    for page in pages:
        res_left, res_right = None, None
        if win_type == 'default':
            # calculate eye features with defined time window
            res_left, res_right = calculate_all_features(page)
            
        elif win_type == 'same-dur':
            page_key = str(page.page_num) + str(page.mw_reported)
            if page_key in win_time:
                win_start, win_end = win_time[page_key]
                # calculate eye features with defined time window
                res_left, res_right = calculate_all_features(page, win_start=win_start, win_end=win_end)
        
        # save current page results
        if res_left is not None:
            # fill in other information (i.e. page, time info etc)
            fill_dict(res_left, page)
            # append the res dict to list
            all_res_left.append(res_left)
        
        if res_right is not None:
            fill_dict(res_right, page)
            all_res_right.append(res_right)

    return pd.DataFrame(all_res_left), pd.DataFrame(all_res_left)


def compute_window_time(pages, q_normal, q_mw):
    
    win_time = {}
    while len(q_mw):
        if len(q_normal):
            page_normal = q_normal.popleft()
            page_mw = q_mw.popleft()
             # Calculate offsets and duration for the mind-wandering window
            mw_onset = page_mw.mw_onset - page_mw.time_start
            mw_offset = page_mw.mw_offset - page_mw.time_start
            mw_dur = page_mw.mw_offset - page_mw.mw_onset

            if page_normal.time_start + mw_offset > page.time_end:
                win_start = random.uniform(page_normal.time_start, page_normal.time_end-mw_dur)
            else:
                win_start = page_normal.time_start + mw_onset
            win_end = win_start + mw_dur

            # save win_start and win_end to the dictionary
            page_key = str(page_normal.page_num) + str(page_normal.mw_reported)
            win_time[page_key] = (win_start, win_end)

            page_key = str(page_mw.page_num) + str(page_mw.mw_reported)
            win_time[page_key] = (page_mw.mw_onset, page_mw.mw_offset)

        else:
            page_mw = q_mw.popleft()
            mw_dur = page_mw.mw_offset - page_mw.mw_onset
            nr_dur = page_mw.mw_onset - page_mw.time_start
            if mw_dur > nr_dur:
                continue
            page_normal = copy.deepcopy(page_mw)
            page_normal.mw_reported = False
            page_normal.mw_onset = np.nan
            page_normal.mw_offset = np.nan
            page_normal.mw_valid = False
            
            win_start = random.uniform(page_normal.time_start, page_normal.mw_onset-mw_dur)
            win_end = win_start + mw_dur
            
            # save win_start and win_end to the dictionary
            page_key = str(page_normal.page_num) + str(page_normal.mw_reported)
            win_time[page_key] = (win_start, win_end)

            page_key = str(page_mw.page_num) + str(page_mw.mw_reported)
            win_time[page_key] = (page_mw.mw_onset, page_mw.mw_offset)

            # insert the "normal" page right before the mind-wandering page
            for index, page in enumerate(pages):
                if page.page_num == page_mw.page_num:
                    pages.insert(index, page_normal)
                    break

    return pages, win_time


def fill_dict(res, page):
    '''
    _summary_

    Args:
        res (_type_): _description_
        page (_type_): _description_
    '''    
    reading_convert = {'history_of_film':'History of Film',
                       'pluto':'Pluto',
                       'serena_williams':'Serena Williams',
                       'the_voynich_manuscript':'The Voynich Manuscript',
                       'prisoners_dilemma':'Prisoners Dilemma'}
    
    res['reading'] = reading_convert[page.reading]
    res['page'] = page.page_number
    res['is_MWreported'] = page.mw_reported
    res['is_MWvalid'] = page.mw_valid
    res['MW_start'] = page.mw_onset
    res['MW_end'] = page.mw_offset
    res['page_start'] = page.time_start
    res['page_end'] = page.time_end
    res['win_start'] = page.win_start
    res['win_end'] = page.win_end
    res['win_dur'] = page.win_dur
    res['task_start'] = page.task_start

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


def save_page_data_to_dict(eye_dict, page, run, reading_convert, win_type):
    '''
    This is a helper function to call analyses funcs and save eye features 
    in page object to a dict. 

    Parameters
    ----------
    eye_dict : dict
        DESCRIPTION. a reference of eye_feature_dict, which is a dict of eye
            features and sample labels that are saved in the .csv file. 
    page : object
        DESCRIPTION. See page.py for more details. 
    run : int
        DESCRIPTION. the run/session index number extracted from input file
    reading_convert : dict
        DESCRIPTION. a dict of reading names.
    win_type : str or np.nan
        DESCRIPTION. a sample label

    Returns
    -------
    None. But the input eye_dict gets modified so that it stores the 
    information of the current page object

    '''
    
    # calculate eye features
    page.calculate_features()
    if page.num_fixations > 0:
        page.regressions_in_word()
        page.normalize_data()
    
    # add data to table (this is a dict that will be saved as a csv)
    eye_dict['run'].append(run)
    eye_dict['reading'].append(reading_convert[page.reading])
    eye_dict['page_num'].append(page.page_number) 
    eye_dict['num_blinks'].append(page.num_blinks) 
    eye_dict['num_fixations'].append(page.num_fixations)
    eye_dict['num_saccades'].append(page.num_saccades)
    eye_dict['mean_pupil_size'].append(page.pupil_mean)
    eye_dict['norm_pupil_size'].append(np.nan)
    eye_dict['pupil_slope'].append(page.pupil_slope)
    eye_dict['mean_blink_duration'].append(page.mean_blink_duration)
    eye_dict['norm_blink_freq'].append(page.norm_blink_freq)
    eye_dict['norm_num_word_fix'].append(page.norm_num_word_fix)
    eye_dict['norm_saccade_count'].append(page.norm_saccade_count)
    eye_dict['norm_in_word_regression'].append(page.norm_in_word_regression)
    eye_dict['norm_out_word_regression'].append(page.norm_out_word_regression)
    eye_dict['zipf_duration_correlation'].append(page.zipf_duration_correlation)
    eye_dict['word_length_duration_correlation'].append(page.word_length_duration_correlation)
    eye_dict['norm_total_viewing'].append(page.norm_total_viewing)
    eye_dict['reported_MW'].append(int(page.mw_reported))
    eye_dict['valid_MW'].append(int(page.valid_mw))
    eye_dict['task_start'].append(page.task_start)
    eye_dict['win_start'].append(page.mw_onset)
    eye_dict['win_end'].append(page.mw_offset)
    eye_dict['win_duration'].append(page.duration)
    eye_dict['win_type'].append(win_type)
    eye_dict['page_start'].append(page.time_start)
    eye_dict['page_end'].append(page.time_end)


def extract_feature_whole(eye_feature_dict_whole, page, run, reading_convert):
    '''
    This function saves analyzed eye features over the whole period into the 
    dict. For normal reading sample, the time window is the whole reading page.
    For mindless reading sample, the time window is the whole reported MW 
    window. Note that reported but invalid (i.e. duration < 5s) samples do not
    get assigned win_type (i.e. win_type = np.nan).
    
     ________________________
    |_______NR sample_______|
     ________________________  ________________
    |_______________________|||___MR sample___|
                             |
                           mw onset

    Parameters
    ----------
    eye_feature_dict_whole : dict
        DESCRIPTION. a reference of a deepcopy of eye_feature_dict, which is 
        a dict of eye features and sample labels that are saved in the .csv file. 
    page : object
        DESCRIPTION. See page.py for more details. 
    run : int
        DESCRIPTION. the run/session index number extracted from input file
    reading_convert : dict
        DESCRIPTION. a dict of reading names.

    Returns
    -------
    None. But the eye_feature_dict_whole dict gets modified so that it stores
    the informatino of the current page object

    '''
    # assign win_type to the current sample
    # NR: normal reading; no reported MW
    # MR: mindless reading; reported MW and valid MW
    # nan: reported MR but invalid MW
    if page.mw_reported:
        page.win_type = -1
        if page.valid_mw:
            win_type = 'MR'
        else:
            win_type = np.nan
    else:
        win_type = 'NR'
    
    # add data to table (this is a dict that will be saved as a csv)
    save_page_data_to_dict(eye_feature_dict_whole, page, run, 
                           reading_convert, win_type)
    

def extract_feature_same(eye_feature_dict_same, page, run, reading_convert, 
                         time_offset=2):
    '''
    This function saves analyzed eye features over the same period into the 
    dict. For normal reading sample, the time window is just before the 
    reported mind-wandering onset. For mindless reading sample, the time 
    window is the normal reported MW window. NR and MR are modifeid so that
    they have the same duration. Note that normal reading page and reported but 
    invalid (i.e. duration < 5s) samples do not get assigned win_type 
    (i.e. win_type = np.nan).
     ________________                                ________________
    |___NR sample___|__time offset__|__time offset__|___MR sample___|
                                    |
                                 mw onset

    Parameters
    ----------
    eye_feature_dict_same : dict
        DESCRIPTION. a reference of a deepcopy of eye_feature_dict, which is 
        a dict of eye features and sample labels that are saved in the .csv file. 
    page : object
        DESCRIPTION. See page.py for more details. 
    run : int
        DESCRIPTION. the run/session index number extracted from input file
    reading_convert : dict
        DESCRIPTION. a dict of reading names.
    time_offset : int, optional
        DESCRIPTION. the offset between NR and MR time windows in seconds. 
            The default is 2.

    Returns
    -------
    None. But the eye_feature_dict_same dict gets modified so that it stores
    the informatino of the current page object

    '''
    # for page with valid reported MW
    if page.valid_mw:
        # create sample for mindless reading
        win_type = 'MR'
        page.win_type = -1
        # subtract time_offset from the onset of MW
        page.mw_onset += time_offset
        # if NR time window is shorter than MR
        # truncate the MR window time to keep them the same duration
        MR_duration = page.mw_offset-page.mw_onset
        NR_duration = page.mw_onset-page.time_start-time_offset*2
        if MR_duration > NR_duration:
            page.mw_offset = page.mw_onset + NR_duration
        # add data to table (this is a dict that will be saved as a csv)
        save_page_data_to_dict(eye_feature_dict_same, page, run, 
                               reading_convert, win_type)
          
        # create sample for normal reading with the same duration
        win_type = 'NR'
        page.win_type = 1
        page.mw_offset = page.mw_onset-time_offset*2
        page.mw_onset = np.max((page.time_start,
                                (page.mw_offset-MR_duration)))
        # add data to table (this is a dict that will be saved as a csv)
        save_page_data_to_dict(eye_feature_dict_same, page, run, 
                               reading_convert, win_type)
    
    # for page without valid reported MW
    else:
        win_type = np.nan
        # add data to table (this is a dict that will be saved as a csv)
        save_page_data_to_dict(eye_feature_dict_same, page, run, 
                               reading_convert, win_type)


def extract_feature_last(eye_feature_dict_last, page, run, reading_convert, 
                         win_duration=5):
    '''
    This function saves analyzed eye features over the last a few seconds on 
    the page into the dict for both normal and mindless reading. Note that 
    reported but invalid (i.e. duration < 5s) samples do not get assigned 
    win_type (i.e. win_type = np.nan).
    
    dur(NR) = dur(MR) = win_duration
     ________________                                
    |___NR sample___|__|
                       |
                page end time
     ________________                                
    |___MR sample___|__|
                       |
                page end time           

    Parameters
    ----------
    eye_feature_dict_last : dict
        DESCRIPTION. a reference of a deepcopy of eye_feature_dict, which is 
        a dict of eye features and sample labels that are saved in the .csv file. 
    page : object
        DESCRIPTION. See page.py for more details. 
    run : int
        DESCRIPTION. the run/session index number extracted from input file
    reading_convert : dict
        DESCRIPTION. a dict of reading names.
    win_duration : int, optional
        DESCRIPTION. the duration of sample time window in seconds. 
            The default is 5.

    Returns
    -------
    None. But the eye_feature_dict_last dict gets modified so that it stores
    the informatino of the current page object

    '''
    # declare time offset to account for gaze not on the monitor
    time_offset = 1
    
    # assign win_type to the current sample
    # NR: normal reading; no reported MW
    # MR: mindless reading; reported MW and valid MW
    # nan: reported MR but invalid MW
    if page.mw_reported:
        win_type = 'MR'
        page.win_type = -1
    else:
        win_type = 'NR'
        # set bool to be true so that eye features will be extracted only from
        # defined time window (i.e. last a few seconds on the page)
        page.mw_reported = True
        page.valid_mw = True
        
    # create sample over last several seconds on the current page
    # page.mw_offset = page.time_end - time_offset
    # page.mw_onset = page.mw_offset - win_duration
    page.mw_offset = page.time_end - time_offset
    page.mw_onset = page.time_start
        
    # add data to table (this is a dict that will be saved as a csv)
    save_page_data_to_dict(eye_feature_dict_last, page, run, 
                           reading_convert, win_type)


                

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


