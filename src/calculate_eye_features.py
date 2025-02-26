# calculate_eye_features.py
#   This module file contains all functions that caluclate eye features

# Created 10/23/2024 by HS

import numpy as np
import pandas as pd
import warnings
warnings.simplefilter("error", RuntimeWarning)
from .utils import truncate_df_by_time

def calculate_all_features(page):
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
    # check input start and end time are valid
    if page.win_start >= page.win_end:
        print('Window start time cannot be later than end time!')
        return None, None
    elif (page.win_start is None) or (page.win_end is None):
        print('Window start/end time is None!')
        return None, None
    
    page.win_dur = page.win_end - page.win_start
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
    # truncate dataframes in the page object for calcuation
    dfSamples, dfFix, dfSacc, dfBlink = truncate_df_by_time(page.dfSamples, page.dfFix, page.dfSacc, page.dfBlink, 
                                                            page.win_start, page.win_end)
    
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
    dfFixL = dfFix[dfFix['eye']=='L']
    dfFixR = dfFix[dfFix['eye']=='R']
    dfFix = dfFix[dfFix['eye']==eye]
    # number of fixations
    fix_num = len(dfFix)
    # Initialize all features as NaN
    fix_word_num, norm_fix_word_num = np.nan, np.nan
    in_word_reg, out_word_reg = np.nan, np.nan
    zipf_fixdur_corr, wordlength_fixdur_corr = np.nan, np.nan
    fix_dispersion = np.nan
    weighted_vergence = np.nan
    norm_total_viewing = np.nan

    # for at least 1 fixation sample
    if fix_num > 0:
        # number of fixations on word
        fix_word_num = np.sum(dfFix['word_len'] > 0)
        norm_fix_word_num = fix_word_num / page.win_dur if fix_word_num > 0 else np.nan
        norm_total_viewing = calculate_total_viewing(dfFix)

    # for at least 2 fixation samples
    if fix_num > 1:
        # Compute additional features if multiple fixations exist
        in_word_reg, out_word_reg = calculate_word_regression(dfFix)
        zipf_fixdur_corr, wordlength_fixdur_corr = calculate_word_fix_corr(dfFix)
        fix_dispersion = calculate_fix_dispersion(dfFix)
        weighted_vergence = calculate_vergence(dfFixL, dfFixR)

    # save to results dict
    res['fix_num'] = fix_num if fix_num > 0 else np.nan
    res['fix_word_num'] = fix_word_num
    res['norm_fix_word_num'] = norm_fix_word_num
    res['norm_in_word_reg'] = in_word_reg
    res['norm_out_word_reg'] = out_word_reg
    res['zipf_fixdur_corr'] = zipf_fixdur_corr
    res['word_length_fixdur_corr'] = wordlength_fixdur_corr
    res['norm_total_viewing'] = norm_total_viewing
    res['fix_dispersion'] = fix_dispersion
    res['weighted_vergence'] = weighted_vergence

    
    # Blink related features
    dfBlink = dfBlink[dfBlink['eye']==eye]
    # number of blinks
    blink_num = len(dfBlink)
    if blink_num > 0:
        # average blink duration
        blink_dur = dfBlink['duration'].mean()
        # blink frequency
        blink_freq = blink_num / page.win_dur
    else:
        blink_num, blink_dur, blink_freq = np.nan, np.nan, np.nan
    # save to results dict
    res['blink_num'] = blink_num
    res['blink_dur'] = blink_dur
    res['blink_freq'] = blink_freq


    # Saccade related features
    dfSacc = dfSacc[dfSacc['eye']==eye]
    # number of saccades
    sacc_num = len(dfSacc)
    # compute mean saccade length and horizontal saccade proportion
    if sacc_num > 0:
        norm_sacc_num = sacc_num / page.win_dur
        sacc_length = calculate_mean_sacc_length(dfSacc)
        horizontal_sacc = calculate_horizontal_sacc_proportion(dfSacc)
    else:
        sacc_num = np.nan
        norm_sacc_num = np.nan
        sacc_length, horizontal_sacc = np.nan, np.nan
    # save to results dict
    res['sacc_num'] = sacc_num
    res['norm_sacc_num'] = norm_sacc_num
    res['sacc_length'] = sacc_length
    res['horizontal_sacc'] = horizontal_sacc


    # Pupil (Samples) related features
    pupil = dfSamples[f'{eye}Pupil']
    # check dataframe size
    if len(pupil) < 2:
        mean_pupil = np.nan
        slope = np.nan
    else:
        # mean pupil size
        mean_pupil = pupil.mean()
        try:
            # pupil slope
            slope, intercept = np.polyfit(np.arange(0,len(pupil)), pupil, 1)
        except np.linalg.LinAlgError as e:
            print(f"Warning: SVD did not converge in Linear Least Squares. Error: {e}")
            slope = np.nan
    
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
    if len(dfFix_word):
        norm_total_viewing = np.mean(dfFix_word['duration'])
    else:
        norm_total_viewing = np.nan
    return norm_total_viewing


def calculate_word_fix_corr(dfFix):
    '''
    Calculates the correlation between word fixation duration and two features: 
    Zipf frequency and word length. The function filters the dataframe to exclude 
    invalid data (e.g., missing values or zero Zipf scores) and then computes 
    Pearson's correlation coefficient for the two correlations.

    Parameters
    ----------
    dfFix : pandas.DataFrame
        A dataframe containing fixation data for words, with columns 'duration' 
        (fixation duration), 'zipf' (Zipf frequency), and 'word_len' (word length).

    Returns
    -------
    zipf_fixdur_corr : float or np.nan
        Pearson correlation coefficient between Zipf score and word fixation duration.
        Returns np.nan if correlation can't be computed (e.g., due to insufficient data or zero variance).
        
    wordlength_fixdur_corr : float or np.nan
        Pearson correlation coefficient between word length and word fixation duration.
        Returns np.nan if correlation can't be computed (e.g., due to insufficient data or zero variance).
        
    Notes
    -----
    - **Zipf score correlation**: Only fixations with a non-zero Zipf score are considered.
    - **Word length correlation**: Only fixations on words with positive word length are considered.
    - If there are fewer than 2 valid data points or if the standard deviation of either variable is zero, the function will return `np.nan` for that correlation.
    '''
    
    if len(dfFix) < 2:
        raise ValueError('dfFix must have at least 2 samples to calculate correlation')

    # Clean data: remove NaN and Inf values in key columns
    dfFix = dfFix.dropna(subset=['duration', 'zipf', 'word_len'])
    dfFix = dfFix[np.isfinite(dfFix['word_len'])]

    # Sensitivity of Zipf frequency
    # Exclude fixations on words that have no Zipf score (zero Zipf)
    dfFix_word = dfFix.loc[dfFix['zipf'] != 0]
    if len(dfFix_word) >= 2 and np.std(dfFix_word['zipf']) > 0 and np.std(dfFix_word['duration']) > 0:
        # Calculate correlation coefficient between Zipf score and fixation duration
        zipf_fixdur_corr = dfFix_word['zipf'].corr(dfFix_word['duration'])
    else:
        zipf_fixdur_corr = np.nan
    
    # Sensitivity of word length
    # Exclude fixations not on valid words (positive word length)
    dfFix_word = dfFix.loc[dfFix['word_len'] > 0]
    if len(dfFix_word) >= 2 and np.std(dfFix_word['word_len']) > 0 and np.std(dfFix_word['duration']) > 0:
        # Calculate correlation coefficient between word length and fixation duration
        wordlength_fixdur_corr = dfFix_word['word_len'].corr(dfFix_word['duration'])
    else:
        wordlength_fixdur_corr = np.nan

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

    if len(dfFix) < 2:
        raise ValueError('dfFix must have at least 2 samples to calculate regression')

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


def calculate_fix_dispersion(dfFix):
    """
    Calculate the fixation dispersion as the root mean square of the distance of each fixation 
    to the average fixation position, but only within a specified time window.
    
    Returns:
        fixation_dispersion (float): The calculated dispersion for the given window.
    """
    
    # Calculate the average x and y fixation positions 
    x_avg = dfFix['xAvg'].mean()
    y_avg = dfFix['yAvg'].mean()

    # Calculate the distance of each fixation to the average position
    #d=√((x2 – x1)² + (y2 – y1)²)
    dist = np.sqrt((x_avg-dfFix['xAvg'])**2 + (y_avg-dfFix['yAvg'])**2)
    
    # Calculate the fixation dispersion (root mean square of the distances)
    fixation_dispersion = np.sqrt(np.mean(dist**2))
    
    return fixation_dispersion


def calculate_mean_sacc_length(dfSacc):
    """
    Calculate the mean saccade length based on start and end positions of saccades
    within a specific window defined by the 'in_win' column.
    
    Returns:
        mean_saccade_length (float): The average length of all saccades in the window.
    """
    # convert df to float 
    dfSacc = dfSacc.apply(pd.to_numeric, errors='coerce')  # Converts invalid values to NaN
    # Calculate the saccade length for each row using the Euclidean distance formula
    sacc_length = np.sqrt((dfSacc['xEnd']-dfSacc['xStart'])**2 + (dfSacc['yEnd']-dfSacc['yStart'])**2)

    return np.mean(sacc_length)


def calculate_horizontal_sacc_proportion(dfSacc):
    """
    Calculate the proportion of horizontal saccades (within 30 degrees of the x-axis).

    Returns:
        horizontal_saccade_proportion (float): Proportion of horizontal saccades.
    """
    # this feature is questionable in our dataset because Eyelink stores the saccade 
    # angle between two fixations as ampDeg in the saccade file. 
    # I observed that there are very few data points exceeding 30 degrees—only a handful in an entire run.
    return np.nan
    

def calculate_vergence(dfFixL, dfFixR):
    """
    Calculate the weighted vergence between two eye fixation datasets (left and right eyes).

    The function merges the two dataframes, dfFixL and dfFixR, by matching rows based on the 'tStart' column
    using the `merge_asof` method. After merging, it calculates the vergence distance at each time point and
    computes a weighted average of the vergence distances based on the fixation durations.

    Parameters:
    ----------
    dfFixL : pd.DataFrame
        A DataFrame containing fixation data for the left eye, with columns such as 'tStart', 'duration',
        'xAvg', and 'yAvg'.
    
    dfFixR : pd.DataFrame
        A DataFrame containing fixation data for the right eye, with columns such as 'tStart', 'duration',
        'xAvg', and 'yAvg'.
    
    Returns:
    -------
    float
        The weighted vergence value, which is the average vergence distance weighted by the fixation duration.
        This value represents the overall vergence between the two eyes based on the fixation data.
    
    Notes:
    ------
    - Only fixations with matching time points (within a tolerance of 10 ms) between the two eyes are considered.
    - Rows with missing ('NaN') fixation position data in either eye are dropped.
    - The vergence distance is computed as the Euclidean distance between the fixation points of the left and right eyes.
    - The weighted vergence is computed as the sum of vergence distances multiplied by the corresponding fixation durations,
      divided by the total fixation duration.
    """
     # Merge using merge_asof on tStart, ensuring we don't lose any fixations
    df_merged = pd.merge_asof(
        dfFixL.sort_values('tStart'), 
        dfFixR.sort_values('tStart'), 
        on='tStart', 
        direction='nearest',  # Get the nearest match for tStart from both sides
        tolerance=10,
        suffixes=('_L', '_R')
    )
    # Drop rows with NaN values
    df_merged = df_merged.dropna(subset=['xAvg_L', 'xAvg_R'])

    if len(df_merged):
        # Compute fixation duration (average of both eyes)
        duration = (df_merged['duration_L'] + df_merged['duration_R']) / 2
        total_dur = duration.sum()

        # Get position info for both eyes
        xAvg_L, yAvg_L = df_merged['xAvg_L'], df_merged['yAvg_L']
        xAvg_R, yAvg_R = df_merged['xAvg_R'], df_merged['yAvg_R']

        # compute vergence distance
        vergence_dist = np.sqrt((xAvg_L-xAvg_R)**2 + (yAvg_L-yAvg_R)**2)
        # compute weighted vergence
        weighted_vergence = np.sum(vergence_dist * duration) / total_dur
    else:
        weighted_vergence = np.nan
    
    return weighted_vergence