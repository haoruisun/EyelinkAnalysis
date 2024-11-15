# calculate_eye_features.py
#   This module file contains all functions that caluclate eye features

# Created 10/23/2024 by HS

import numpy as np
import pandas as pd
from .utils import truncate_df_by_time

def calculate_all_features(page, win_start=None, win_end=None):
    '''
    _summary_

    Args:
        page (_type_): _description_
        win_start (_type_, optional): _description_. Defaults to None.
        win_end (_type_, optional): _description_. Defaults to None.

    Raises:
        ValueError: _description_
        ValueError: _description_

    Returns:
        _type_: _description_
    '''    
    needs_truncate = True
    # check if both start and end are Nones or real values
    if (win_start is None) != (win_end is None):
        raise ValueError('Window start and end are not consistent!')
    # for nonspecified start and end, use either page or MW episode time
    elif (win_start is None) and (win_end is None):
        if page.mw_reported:
            win_start = page.mw_onset
            win_end = page.mw_offset
        else:
            needs_truncate = False
            win_start = page.time_start
            win_end = page.time_end
    # for input start and end time, check they are valid
    elif win_start >= win_end:
        raise ValueError('Window start time cannot be later than end time!')
    
    # truncate the dataframes by time
    if needs_truncate:
        truncate_df_by_time(page, win_start, win_end)
    
    # calculate single eye feature
    res_left = calculate_single_eye(page, 'L')
    res_right = calculate_single_eye(page, 'R')

    return res_left, res_right


def calculate_single_eye(page, eye):
    '''
    _summary_

    Args:
        page (_type_): _description_
        eye (_type_): _description_

    Returns:
        _type_: _description_
    '''    
    # check mono or bino recording mode
    dfSamples = page.dfSamples
    # Calculate the number of empty rows in the specified column
    empty_rows = dfSamples[f'{eye}Pupil'].isnull().sum()
    # Check if empty rows in the column exceed one-tenth of the total rows
    if empty_rows > 1/10 * len(dfSamples):
        # return None for eye that has non-readings
        return None
    
    # declare dict to hold feature results
    res = {}
    # specify which eye
    res['eye'] = eye

    # Fixation related features
    dfFix = page.dfFix
    dfFix = dfFix[dfFix['eye']==eye]
    # number of fixations
    fix_num = len(dfFix)
    # number of fixations on word
    fix_word_num = np.sum(dfFix['word_len'] > 0)
    # call functions to calculate other features
    in_word_reg, out_word_reg = calculate_word_regression(dfFix)
    zipf_fixdur_corr, wordlength_fixdur_corr = calculate_word_fix_corr(dfFix)
    norm_total_viewing = calculate_total_viewing(dfFix)
    # save to results dict
    res['fix_num'] = fix_num
    res['fix_word_num'] = fix_word_num
    res['norm_in_word_reg'] = in_word_reg
    res['norm_out_word_reg'] = out_word_reg
    res['zipf_fixdur_corr'] = zipf_fixdur_corr
    res['word_length_fixdur_corr'] = wordlength_fixdur_corr
    res['norm_total_viewing'] = norm_total_viewing

    
    # Blink related features
    dfBlink = page.dfBlink
    dfBlink = dfBlink[dfBlink['eye']==eye]
    # number of blinks
    blink_num = len(dfBlink)
    # average blink duration
    blink_dur = dfBlink['duration'].mean()
    # blink frequency
    blink_freq = blink_num / page.win_dur if blink_num > 0 else np.nan
    # save to results dict
    res['blink_num'] = blink_num
    res['blink_dur'] = blink_dur
    res['blink_freq'] = blink_freq


    # Saccade related features
    dfSacc = page.dfSacc
    dfSacc = dfSacc[dfSacc['eye']==eye]
    # number of saccades
    sac_num = len(dfSacc)
    norm_sac_num = sac_num / page.win_dur
    # save to results dict
    res['sac_num'] = sac_num
    res['norm_sac_num'] = norm_sac_num


    # Pupil (Samples) related features
    dfSamples = page.dfSamples
    pupil = dfSamples[f'{eye}Pupil']
    # mean pupil size
    mean_pupil = pupil.mean()
    # pupil slope
    slope, intercept = np.polyfit(np.arange(0,len(pupil)), pupil, 1)
    # save to results dict
    res['mean_pupil'] = mean_pupil
    res['pupil_slope'] = slope

    return res



def calculate_total_viewing(dfFix):
    '''
    _summary_

    Args:
        dfFix (_type_): _description_

    Returns:
        _type_: _description_
    '''    
    # drop fixations that are not on a word
    dfFix_word = dfFix.loc[dfFix['word_len'] > 0]
    # compuate the normalized total viewing time
    norm_total_viewing = np.mean(dfFix_word['duration'])
    return norm_total_viewing


def calculate_word_fix_corr(dfFix):
    '''
    _summary_

    Args:
        dfFix (_type_): _description_
    '''    
     # sensitivity of word frequency
    # exclude fixations on the word that has no zipf score
    dfFix_word = dfFix.loc[dfFix['zipf'] != 0]
    # calc correlation coeffiecient r between zipf score and word fixation duration
    zipf_fixdur_corr = dfFix_word['zipf'].corr(dfFix_word['duration'])
    
    # sensitivity of word length
    # exclude fixations not on the word
    dfFix_word = dfFix.loc[dfFix['word_len'] > 0]
    # calc correlation coeffiecient r between word length and word fixation duration
    wordlength_fixdur_corr = dfFix_word['word_len'].corr(dfFix_word['duration'])

    return zipf_fixdur_corr, wordlength_fixdur_corr


def calculate_word_regression(dfFix):
    '''
    This function finds fixations that doesn't follow a normal reading pattern (left -> right)
    in_word_regression - the following fixations that are with-in the word
    out_word_regression - the following fixations that are on previous words

    Args:
        df_fixations (df): _description_

    Returns:
        in_word_reg (float): in word regression rate normalized over all fixations
        out_word_reg (float): out word regression rate normalized over all fixations
    '''    
    # declare empty list to store in and out word regression words
    in_word_regression = []
    out_word_regression = []
   
    # use the first fixation word as reference
    prev_fix = dfFix.iloc[0]
    
    # loop through all the fiaxtions and compare with 
    for _, fix in dfFix[1:].iterrows():
        
        # get the index of previous fixation word
        prev_index = prev_fix['fixed_word_index']
        
        # get the index of current fixation word
        index = fix['fixed_word_index']
        
        # check for in-word regression
        if (prev_index == index):
            in_word_regression.append(fix)
        # check for out-word regression
        elif (index < prev_index):
            out_word_regression.append(fix)
        
        # update the previous fixation word
        prev_fix = fix

    # normalize the features over all fixations
    in_word_reg = len(in_word_regression) / len(dfFix)
    out_word_reg = len(out_word_regression) / len(dfFix)
    return in_word_reg, out_word_reg

