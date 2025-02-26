#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module file contains all function used for quality check. 

Created on 02/14/2023 by HS
Updated 01/23/2024 by HS - append three different eye features files
                               separately and barplot them together
Updated 2/20/24 by HS - QA on comprehension questionss
Updated 2/27/24 by HS - add z-scored corr coefficien to group df
Updated 5/1/24 by HS - estimate gamma distribution para for MW duration

"""

# %% Packages
import os
import re
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
from statannot import add_stat_annotation
from scipy import stats

# %% Parameters and Variables
# define root path relative to current working directory
path = '../../../../Data/'
#path = '../../Data/'
subfolders = [f.path for f in os.scandir(path) if f.is_dir()]

# declare empty dict to store reading time
readings = ['History of Film', 'Pluto', 'Prisoners Dilemma', 
            'Serena Williams', 'The Voynich Manuscript']
time = {reading_name:[] for reading_name in readings}

# declare eye feature column names for plotting
cols = ['norm_blink_freq', 'pupil_slope',
        'mean_pupil_size', 'mean_blink_duration', 'norm_pupil_size',
        'norm_num_word_fix', 'norm_saccade_count', 'norm_in_word_regression',
        'norm_out_word_regression', 'zscored_zipf_duration_correlation',
        'zipf_duration_correlation', 'norm_total_viewing',
        'zscored_word_length_duration_correlation',
        'word_length_duration_correlation']


# %% Functions
def append_files(path, file_pattern='*[mono|R]_features*', 
                 file_name='group_R_features.csv'):
    '''
    Append subject files (eye/behavior) and save as a group .csv file

    Parameters
    ----------
    path : string
        DESCRIPTION. The relative path to the data folder
    file_pattern : string, optional
        DESCRIPTION. The pattern for files being appended. 
        The default is '*[mono|R]_features*' for eye feature files.
    file_name : string, optional
        DESCRIPTION. the group file name to save
        The default is 'group_R_features.csv'.

    Returns
    -------
    df_group : dataframe
        DESCRIPTION. The group dataframe 

    '''
    # grab every sub folders under the input path
    sub_folders = [f.path for f in os.scandir(path) if f.is_dir()]
    # define group dataframe
    df_group = pd.DataFrame()
    
    # loop through every folder
    for folder_path in sub_folders:
        try: 
            # extract subject id
            subject_id = re.findall(r's[0-9]+', folder_path)[0]
            # extract tracking and behavior features
            file_path = glob(os.path.join(folder_path, file_pattern))
            # read in the subject csv
            df_ind = pd.read_csv(file_path[0])
            # add subject id columne
            df_ind['sub_id'] = subject_id
            # append to the group dataframe
            df_group = pd.concat([df_group, df_ind], ignore_index=True)
        except:
            continue
    
    # save and return the group dataframe
    df_group = df_group.loc[:, ~df_group.columns.str.match('Unnamed')]
    
    # calculate z-score of two correlation coefficient columns
    # zipf_duration_correlation and word_length_duration_correlation
    for col_name in ['zipf_fixdur_corr', 'word_length_fixdur_corr']:
        # get index
        index = df_group.columns.get_loc(col_name)
        # extract the column values
        col = df_group[col_name]
        # compute the z-score
        z_col = stats.zscore(col, nan_policy='omit')
        # insert into the dataframe
        df_group.insert(index, f'zscored_{col_name}', z_col)
    
    df_group.to_csv(f'{path}{file_name}')
    return df_group
        

def plot_eye_features_group(df_whole, df_same, df_last, 
                            path='../../../Data/QualityCheckPlot/EyeFeatures/Group/'):
    
    # loop through features of interests 
    for col in cols:
        plt.figure(figsize=(15, 15))
        
        # whole group
        df = df_whole
        # bar plot
        plt.subplot(2, 3, 1)
        ax = sns.barplot(df, x='win_type', y=col)
        add_stat_annotation(ax, data=df, x='win_type', y=col, box_pairs=[('MR', 'NR')],
                            test='Mann-Whitney', text_format='full', loc='outside', verbose=2)
        plt.grid()
        plt.ylabel('whole')
        # violin plot
        plt.subplot(2, 3, 4)
        sns.violinplot(df, x='win_type', y=col, inner='stick')
        plt.grid()
        
        # same group
        df = df_same
        # bar plot
        plt.subplot(2, 3, 2)
        ax = sns.barplot(df, x='win_type', y=col)
        add_stat_annotation(ax, data=df, x='win_type', y=col, box_pairs=[('MR', 'NR')],
                            test='Mann-Whitney', text_format='full', loc='outside', verbose=2)
        plt.grid()
        plt.ylabel('same')
        # violin plot
        plt.subplot(2, 3, 5)
        sns.violinplot(df, x='win_type', y=col, inner='stick')
        plt.grid()
        
        # last group
        df = df_last
        # bar plot
        plt.subplot(2, 3, 3)
        ax = sns.barplot(df, x='win_type', y=col)
        add_stat_annotation(ax, data=df, x='win_type', y=col, box_pairs=[('MR', 'NR')],
                            test='Mann-Whitney', text_format='full', loc='outside', verbose=2)
        plt.grid()
        plt.ylabel('last')
        # violin plot
        plt.subplot(2, 3, 6)
        sns.violinplot(df, x='win_type', y=col, inner='stick')
        plt.grid()
        
        plt.suptitle(col)
        plt.tight_layout()
        plt.savefig(f'{path}{col}_group.png')
        
        



def plot_eye_features_bar(df_whole, df_same, df_last, 
                      path='../../../Data/QualityCheckPlot/EyeFeatures/'):
    '''
    This function plots the eye features against reported mind-wandering as a
    bar plot. Call append_files() first to get group dataset. 

    Parameters
    ----------
    df : dataframe
        DESCRIPTION. The dataframe that contains all subjects information
    path : string, optional
        DESCRIPTION. The relative path to save the plot
        The default is '../../../Data/QualityCheckPlot/'.

    Returns
    -------
    None.

    '''
    # generate the pairs for t-test
    subs = np.unique(df_whole['sub_id'])
    pairs = [((sub, 'MR'), (sub, 'NR')) for sub in subs]
    
    
    # loop through features of interests 
    for col in df_whole.columns[6:-8]:
        plt.figure(figsize=(12, 18))
        
        # barplot for individuals
        df = df_whole
        plt.subplot(3, 2, 1)
        ax = sns.barplot(df, x='sub_id', y=col, hue='win_type')
        add_stat_annotation(ax, data=df, x='sub_id', y=col, hue='win_type', 
                            box_pairs=pairs, test='Mann-Whitney', 
                            text_format='star', loc='outside', verbose=2)
        plt.ylabel('whole')
        plt.grid()
        #plt.title('whole individual')
        # barplot for the group
        plt.subplot(3, 2, 2)
        ax = sns.barplot(df, x='win_type', y=col)
        add_stat_annotation(ax, data=df, x='win_type', y=col, box_pairs=[('MR', 'NR')],
                            test='Mann-Whitney', text_format='full', loc='outside', verbose=2)
        plt.grid()
        plt.suptitle(col)
        #plt.title('whole group')
        
        # barplot for individuals
        df = df_same
        plt.subplot(3, 2, 3)
        ax = sns.barplot(df, x='sub_id', y=col, hue='win_type')
        add_stat_annotation(ax, data=df, x='sub_id', y=col, hue='win_type', 
                            box_pairs=pairs, test='Mann-Whitney', 
                            text_format='star', loc='outside', verbose=2)
        plt.ylabel('same')
        plt.grid()
        #plt.title('same individual')
        # barplot for the group
        plt.subplot(3, 2, 4)
        ax = sns.barplot(df, x='win_type', y=col)
        add_stat_annotation(ax, data=df, x='win_type', y=col, box_pairs=[('MR', 'NR')],
                            test='Mann-Whitney', text_format='full', loc='outside', verbose=2)
        plt.grid()
        plt.suptitle(col)
        #plt.title('same group')
        
        # barplot for individuals
        df = df_last
        plt.subplot(3, 2, 5)
        ax = sns.barplot(df, x='sub_id', y=col, hue='win_type')
        add_stat_annotation(ax, data=df, x='sub_id', y=col, hue='win_type', 
                            box_pairs=pairs, test='Mann-Whitney', 
                            text_format='star', loc='outside', verbose=2)
        plt.ylabel('last')
        plt.grid()
        #plt.title('last individual')
        # barplot for the group
        plt.subplot(3, 2, 6)
        ax = sns.barplot(df, x='win_type', y=col)
        add_stat_annotation(ax, data=df, x='win_type', y=col, box_pairs=[('MR', 'NR')],
                            test='Mann-Whitney', text_format='full', loc='outside', verbose=2)
        plt.grid()
        plt.suptitle(col)
        #plt.title('last group')
        
        
        plt.tight_layout()
        plt.savefig(f'{path}{col}_barplot.png')
    return



def plot_eye_features_box(df_whole, df_same, df_last, 
                      path='../../../Data/QualityCheckPlot/EyeFeatures/'):
    '''
    This function plots the eye features against reported mind-wandering as a
    box plot. Call append_files() first to get group dataset. 

    Parameters
    ----------
    df : dataframe
        DESCRIPTION. The dataframe that contains all subjects information
    path : string, optional
        DESCRIPTION. The relative path to save the plot
        The default is '../../../Data/QualityCheckPlot/'.

    Returns
    -------
    None.

    '''
    # loop through features of interests 
    for col in df_whole.columns[6:-10]:
        plt.figure(figsize=(12, 18))
        
        # barplot for individuals
        df = df_whole
        plt.subplot(3, 2, 1)
        sns.boxplot(df, x='sub_id', y=col, hue='win_type')
        plt.grid()
        plt.title('whole individual')
        # barplot for the group
        plt.subplot(3, 2, 2)
        ax = sns.boxplot(df, x='win_type', y=col)
        add_stat_annotation(ax, data=df, x='win_type', y=col, box_pairs=[('MR', 'NR')],
                            test='Mann-Whitney', text_format='full', loc='outside', verbose=2)
        plt.grid()
        plt.suptitle(col)
        plt.title('whole group')
        
        # barplot for individuals
        df = df_same
        plt.subplot(3, 2, 3)
        sns.boxplot(df, x='sub_id', y=col, hue='win_type')
        plt.grid()
        plt.title('same individual')
        # barplot for the group
        plt.subplot(3, 2, 4)
        ax = sns.boxplot(df, x='win_type', y=col)
        add_stat_annotation(ax, data=df, x='win_type', y=col, box_pairs=[('MR', 'NR')],
                            test='Mann-Whitney', text_format='star', loc='outside', verbose=2)
        plt.grid()
        plt.suptitle(col)
        plt.title('same group')
        
        # barplot for individuals
        df = df_last
        plt.subplot(3, 2, 5)
        sns.boxplot(df, x='sub_id', y=col, hue='win_type')
        plt.grid()
        plt.title('last individual')
        # barplot for the group
        plt.subplot(3, 2, 6)
        ax = sns.boxplot(df, x='win_type', y=col)
        add_stat_annotation(ax, data=df, x='win_type', y=col, box_pairs=[('MR', 'NR')],
                            test='Mann-Whitney', text_format='full', loc='outside', verbose=2)
        plt.grid()
        plt.suptitle(col)
        plt.title('last group')
        
        
        plt.tight_layout()
        plt.savefig(f'{path}{col}_boxplot.png')
    return


def plot_eye_features_violin(df_whole, df_same, df_last, 
                             path='../../../Data/QualityCheckPlot/EyeFeatures/'):

    # loop through features of interests 
    for col in df_whole.columns[6:-10]:
        plt.figure(figsize=(12, 18))
        
        # barplot for individuals
        df = df_whole
        plt.subplot(3, 2, 1)
        sns.violinplot(df, x='sub_id', y=col, hue='win_type', split=True, inner='stick')
        plt.grid()
        plt.title('whole individual')
        # barplot for the group
        plt.subplot(3, 2, 2)
        sns.violinplot(df, x='win_type', y=col, inner='stick')
        plt.grid()
        plt.suptitle(col)
        plt.title('whole group')
        
        # barplot for individuals
        df = df_same
        plt.subplot(3, 2, 3)
        sns.violinplot(df, x='sub_id', y=col, hue='win_type', split=True, inner='stick')
        plt.grid()
        plt.title('same individual')
        # barplot for the group
        plt.subplot(3, 2, 4)
        sns.violinplot(df, x='win_type', y=col, inner='stick')
        plt.grid()
        plt.suptitle(col)
        plt.title('same group')
        
        # barplot for individuals
        df = df_last
        plt.subplot(3, 2, 5)
        sns.violinplot(df, x='sub_id', y=col, hue='win_type', split=True, inner='stick')
        plt.grid()
        plt.title('last individual')
        # barplot for the group
        plt.subplot(3, 2, 6)
        sns.violinplot(df, x='win_type', y=col, inner='stick')
        plt.grid()
        plt.suptitle(col)
        plt.title('last group')
        
        
        plt.tight_layout()
        plt.savefig(f'{path}{col}_violinplot.png')
    return


def plot_eye_features_win(df, path='../../../Data/QualityCheckPlot/'):
    '''
    This function plots the eye features against reported mind-wandering as a
    bar plot. Call append_files() first to get group dataset. 

    Parameters
    ----------
    df : dataframe
        DESCRIPTION. The dataframe that contains all subjects information
    path : string, optional
        DESCRIPTION. The relative path to save the plot
        The default is '../../../Data/QualityCheckPlot/'.

    Returns
    -------
    None.

    '''
    # loop through features of interests 
    for col in df.columns[6:-10]:
        plt.figure(figsize=(12, 6))
        # barplot for individuals
        plt.subplot(1, 2, 1)
        sns.barplot(df, x='sub_id', y=col, hue='win_type')
        plt.grid()
        # barplot for the group
        plt.subplot(1, 2, 2)
        sns.barplot(df, x='win_type', y=col)
        plt.grid()
        plt.suptitle(col)
        plt.tight_layout()
        plt.savefig(f'{path}{col}_barplot.png')
    return


def extract_info(data_file):
    '''
    This function extracts information from the PsychoPy log file and returns
    them in a dataframe.

    Columns for the returned dataframe:
        page: page number of current row
        reading: reading material (mm/sb)
        reading_time: currnt page reading time in seconds
        error: error type of current row. Default means no implanted error
        key_response: recorded key response. NaN means reader spent the maximum
                     time (50s) on this page
        mouse_error: position of recorded click on errors
        slider_task: recorded value for task slider
        slider_detailed: recorded value for detailed slider
        slider_words: recorded value for words slider
        slider_emotion: recorded value for emotion slider

    Parameters
    ----------
    data_file : string
        DESCRIPTION. The absolute path of PsychoPy log file (.csv)

    Returns
    -------
    DataFrame
        DESCRIPTION. The dataframe that contians desired information from the
                     log file.

    '''
    # load csv file
    df = pd.read_csv(data_file)
    # compute the valid indices based on the error type
    error_type = df['error_type']
    valid_indices = np.where(pd.isnull(error_type) == False)[0]

    # extract the desired information and store them in Series
    error_type = error_type[valid_indices].reset_index(drop=True)
    key_resp = df['key_resp.keys'][valid_indices].reset_index(drop=True)
    reading_time = df['key_resp.rt'][valid_indices].reset_index(drop=True)

    # slider values
    slider_task = df['slider_task.response'][valid_indices].reset_index(drop=True)
    slider_detailed = df['slider_detailed.response'][valid_indices].reset_index(drop=True)
    slider_words = df['slider_words.response'][valid_indices].reset_index(drop=True)
    slider_emotion = df['slider_emotion.response'][valid_indices].reset_index(drop=True)

    # page and reading time
    page = df['page'][valid_indices].reset_index(drop=True)
    reading = df['reading'][valid_indices].reset_index(drop=True)

    # generate the frame with extracted Series
    frame = {'page': page, 'reading': reading, 'reading_time': reading_time,
             'error': error_type, 'key_response': key_resp,
             'slider_task': slider_task,
             'slider_detailed': slider_detailed, 'slider_words': slider_words,
             'slider_emotion': slider_emotion}

    return pd.DataFrame(frame)



def create_cq_df(path):
    '''
    Finds all PsychoPy csv files, extracts answers and subject responses to
    the comprehension questions, and returns the info in a dataframe. 

    Parameters
    ----------
    path : string
        DESCRIPTION. The relative path to the data folder

    Returns
    -------
    df_group : dataframe
        DESCRIPTION. Contains info related to comprehension questions. 
        Columns: reading / answers / responses / subject ID

    '''
    # grab every sub folders under the input path
    sub_folders = [f.path for f in os.scandir(path) if f.is_dir()]
    # define group dataframe
    df_group = pd.DataFrame()

    # loop through every folder
    for folder_path in sub_folders:
        try: 
            # extract subject id
            subject_id = re.findall(r's[0-9]+', folder_path)[0]
            # extract tracking and behavior features
            file_path = glob(os.path.join(folder_path, '**/*R[0-9]_MindlessReading_*.csv'))
            
            # for each psychopy csv file
            for file in file_path: 
                # read in the subject csv
                df_ind = pa.extract_cq_info(file)
                # add subject id columne
                df_ind['sub_id'] = subject_id
                # append to the group dataframe
                df_group = pd.concat([df_group, df_ind], ignore_index=True)
        except:
            continue
    
    return df_group


def compute_cq_correctness(df, path_to_save='../../../Data/QualityCheckPlot/CQ/'):
    '''
    Calculates the correctness of subject responses after excluding 'I don't 
    know' responses and plots results in a barplot.

    Parameters
    ----------
    df : dataframe
        DESCRIPTION. Contains info related to comprehension questions. 
        Columns: reading / answers / responses / subject ID
    path_to_save : string, optional
        DESCRIPTION. path used to save the figure
            The default is '../../../Data/QualityCheckPlot/CQ/'.

    Returns
    -------
    None. Results figure saved in corresponding directory

    '''
    # get unique subject IDs and reading names
    sub_ids = np.unique(df['sub_id'])
    readings = np.unique(df['reading'])

    # throw out background questions
    mask = df['answers']=='None'    
    df = df.loc[~mask]
    # throw out 'I don't know' responses
    mask = df['responses']==5
    df = df.loc[~mask]

    # create the frame dict
    frame = {'sub_id':[], 'correctness':[], 'reading':[]}

    # loop through each subject and reading
    for sub_id in sub_ids:
        for reading in readings:
            # get the corresponding dataframe
            mask = (df['sub_id']==sub_id) & (df['reading']==reading)
            cur_df = df[mask]
            # create the results list with 1 if they answered correctly
            results = [1 if int(ans)==resp else 0 for ans,resp in zip(cur_df['answers'],
                                                                      cur_df['responses'])]
            # compute the correctness
            score = np.mean(results)
            # append results into the list in dict
            frame['sub_id'].append(sub_id)
            frame['correctness'].append(score)
            frame['reading'].append(reading)
    
    # create the results dataframe
    score_df = pd.DataFrame(frame)

    # plot results
    plt.figure(figsize=(12, 12))
    sns.barplot(score_df, x='sub_id', y='correctness', hue='reading')
    plt.axhline(y=0.25, color='r', label='random chance(0.25)')
    plt.grid()
    plt.ylim(0, 1.2)
    plt.legend()
    plt.title('Correctness')
    plt.savefig(f'{path_to_save}correctness.png')
    
    
def compute_cq_attempt_rate(df, path_to_save='../../../Data/QualityCheckPlot/CQ/'):
    '''
    Calculates the percentage of subject responding 'I don't know' over all
    questions. 

    Parameters
    ----------
    df : dataframe
        DESCRIPTION. Contains info related to comprehension questions. 
        Columns: reading / answers / responses / subject ID
    path_to_save : string, optional
        DESCRIPTION. path used to save the figure
            The default is '../../../Data/QualityCheckPlot/CQ/'.

    Returns
    -------
    None. Results figure saved in corresponding directory

    '''
    # get unique subject IDs and reading names
    sub_ids = np.unique(df['sub_id'])
    readings = np.unique(df['reading'])

    # throw out background questions
    mask = df['answers']=='None'    
    df = df.loc[~mask]

    # create the frame dict
    frame = {'sub_id':[], 'rate':[], 'reading':[]}
    
    # loop through each subject and reading
    for sub_id in sub_ids:
        for reading in readings:
            # get the corresponding dataframe
            mask = (df['sub_id']==sub_id) & (df['reading']==reading)
            cur_df = df[mask]
            
            # compute the rate
            rate = np.mean(cur_df['responses']==5)
            # append results into the list in dict
            frame['sub_id'].append(sub_id)
            frame['rate'].append(rate)
            frame['reading'].append(reading)
    
    # create the results dataframe
    rate_df = pd.DataFrame(frame)

    # plot results
    plt.figure(figsize=(12, 12))
    sns.barplot(rate_df, x='sub_id', y='rate', hue='reading')
    plt.grid()
    plt.ylim(0, 0.7)
    plt.title('Not Attempt Rate')
    plt.savefig(f'{path_to_save}attempt_rate.png')



def extract_background_info(df, path_to_save='../../../Data/QualityCheckPlot/CQ/'):
    '''
    Extracts subject responses to the background question (understandability
    and prior knowledge) and plots results in a barplot

    Parameters
    ----------
    df : dataframe
        DESCRIPTION. Contains info related to comprehension questions. 
        Columns: reading / answers / responses / subject ID
    path_to_save : string, optional
        DESCRIPTION. path used to save the figure
            The default is '../../../Data/QualityCheckPlot/CQ/'.

    Returns
    -------
    None. Results figure saved in corresponding directory

    '''
    # get background info
    mask = df['answers']=='None'    
    df = df.loc[mask]
    # separate into understandability (every first row) and prior knowledge (every second row)
    understand_df = df.iloc[::2]
    pk_df = df.iloc[1::2]
    
    # plot results
    plt.figure(figsize=(12, 16))
    # understandability
    plt.subplot(2, 1, 1)
    sns.barplot(understand_df, x='sub_id', y='responses', hue='reading')
    plt.ylabel('  Strongly Disagree   Disagree   Neutral   Agree   Strongly Agree', size=8)
    plt.title('Understandability')
    plt.grid()
    # prior knowledge
    plt.subplot(2, 1, 2)
    sns.barplot(pk_df, x='sub_id', y='responses', hue='reading')
    plt.ylabel('  Strongly Disagree   Disagree   Neutral   Agree   Strongly Agree', size=8)
    plt.title('Prior Knowledge')
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'{path_to_save}background_info.png')
    

def mw_duration_hist(df, path_to_save='../../../Data/QualityCheckPlot/CQ/'):
    
    mask = df['win_end'] > df['win_start']
    df = df[mask]
    avg = df['win_duration'].mean()
    med = df['win_duration'].median()
    var = df['win_duration'].var()
    
    # plot individual duration
    plt.figure()
    sns.histplot(data=df, x='win_duration', hue='sub_id', element="poly")
    plt.title(f'MW Duration (mean:{avg:.2f}, median:{med:.2f}, var:{var:.2f})')
    plt.xlabel('reported mind-wandering duration (s)')
    plt.savefig(f'{path_to_save}mw_duration_ind_hist.png')
    
    # plot group duration
    plt.figure()
    sns.histplot(data=df, x='win_duration', kde=True)
    plt.title(f'MW Duration (mean:{avg:.2f}, median:{med:.2f}, var:{var:.2f}))')
    plt.xlabel('reported mind-wandering duration (s)')
    plt.savefig(f'{path_to_save}mw_duration_hist.png')


def plot_subject_pupiltrace(path):
    
    path_to_save='../../../Data/QualityCheckPlot/Pupil/SubjectPupil/'
    # grab every sub folders under the input path
    sub_folders = [f.path for f in os.scandir(path) if f.is_dir()]

    # loop through every folder
    for folder_path in sub_folders:
        try:
            # extract subject id
            sub_id = re.findall(r's[0-9]+', folder_path)[0]
            # extract pupil trace
            file_path = glob(os.path.join(folder_path, '*_PupilTrace.csv'))
            # read in dataframe
            df = pd.read_csv(file_path[0])
                    
            # check mono or bino recording
            if df.columns[-1][0] in ['L', 'R']:
                eye = 'R'
            else:
                eye = 'mono'
        
            # extract specific epoch type from the dataframe
            onset_col = []
            offset_col = []
            ctr_col = []
            for col in df.columns:
                if f'{eye}_' in col:
                    for epoch_type, epoch_list in zip(['onset', 'offset', 'control'],
                                                      [onset_col, offset_col, ctr_col]):
                        if epoch_type in col:
                            epoch_list.append(col)
        
            # generate sparate dataframe
            df_onset = df[onset_col]
            df_offset = df[offset_col]
            df_ctr = df[ctr_col]
        
            plt.figure(figsize=(12, 12))
            plt.subplot(3, 1, 1)
            sns.lineplot(data=df_onset)
            plt.ylabel('Pupil Change (AU)')
            plt.xlabel('time (ms)')
            plt.grid()
            plt.title(f'{sub_id} Pupil Trace after Onset')
        
            plt.subplot(3, 1, 2)
            sns.lineplot(data=df_offset)
            plt.ylabel('Pupil Change (AU)')
            plt.xlabel('time (ms)')
            plt.grid()
            plt.title(f'{sub_id} Pupil Trace before Offset')
        
            plt.subplot(3, 1, 3)
            sns.lineplot(data=df_ctr)
            plt.ylabel('Pupil Change (AU)')
            plt.xlabel('time (ms)')
            plt.grid()
            plt.title(f'{sub_id} Pupil Trace before Onset (Control)')
            plt.tight_layout()
            plt.savefig(f'{folder_path}/plots/{eye}_PupilTrace.png')
            plt.savefig(f'{path_to_save}{sub_id}_{eye}_PupilTrace.png')
        except:
            continue


# %% Append and save group eye features 
# append subject eye features file
# whole: the timewindow is the whole period for MR and NR
#path = '../../../Data/'
path = '../../Data/'
df_whole = append_files(path, file_pattern='*[mono|R]_features_whole*',
                        file_name='group_R_features_whole.csv')
df_whole = df_whole[df_whole['num_fixations']>0]
df_same = append_files(path, file_pattern='*[mono|R]_features_same*',
                        file_name='group_R_features_same.csv')
df_same = df_same[df_same['num_fixations']>0]
df_last = append_files(path, file_pattern='*[mono|R]_features_last*',
                        file_name='group_R_features_last.csv')
df_last = df_last[df_last['num_fixations']>0]

# %% Append and save group eye features 
path = '../../../../Data/'
df = append_files(path, file_pattern='*[mono|R]_features_same-dur*',
                  file_name='group_R_features_same-dur.csv')


#%% Append default dataset
path = '../../../../Data/'
df = append_files(path, file_pattern='*[mono|R]_features_default*',
                  file_name='group_R_features_default.csv')

#%% Reading time hist
df = df[~df['is_MWreported']]
plt.figure()
df['reading_time'] = df['page_end'] - df['page_start']
avg = df['reading_time'].mean()
med = df['reading_time'].median()
sns.histplot(data=df, x='reading_time', color='gray')
plt.grid()
plt.xlabel('Time(s)')
plt.title(f'Reading Time\nMean:{avg:.2f}, Median:{med:.2f}')

# %% Load group eye features
#file_path = '../../../Data/group_R_features_whole.csv'
file_path = '../../Data/group_R_features_whole.csv'
df_whole = pd.read_csv(file_path)


# %% QA on MW Duration
mw_duration_hist(df_whole)


# %% Delay between offset and self-report
df = df_whole
delay = df['page_end'] - df['win_end']
plt.figure()
sns.histplot(delay)
plt.xlabel('Time(s)')
plt.grid()
plt.suptitle('Time Delay Between Offset and Self-Report')
plt.title(f'Mean: {np.nanmean(delay): .2f}   Median: {np.nanmedian(delay):.2f}')


# %% Delay between page start and onset
df = df_whole
delay = df['win_start'] - df['page_start']
plt.figure()
sns.histplot(delay)
plt.xlabel('Time(s)')
plt.grid()
plt.suptitle('Time Between Page-Start and Onset')
plt.title(f'Mean: {np.nanmean(delay): .2f}   Median: {np.nanmedian(delay):.2f}')


# %% Gamma distribution
# get the MW duration data
df = df_whole
mask = df['win_end'] > df['win_start']
df = df[mask]
dur = df['win_duration']

# calculate stats
avg = dur.mean()
med = dur.median()
var = dur.var()

# # estimate parameters for gamma distribution
# a, c, loc, scale = stats.gengamma.fit(dur)
# # create a new random variable using estimated parameters
# gamma_rv = stats.gengamma.rvs(a, c, loc, scale, size=len(dur))

# # plot results
# plt.figure()
# sns.histplot(dur, kde=True, label='MW duration')
# sns.histplot(gamma_rv, kde=True, label=f'Gamma RV alpha:{a:.2f} beta:{1/scale:.2f}')
# plt.legend()
# plt.title(f'MW Duration (mean:{avg:.2f}, median:{med:.2f}, var:{var:.2f}))')
# plt.xlabel('reported mind-wandering duration (s)')

# estimate parameters for gamma distribution
shape, loc, scale = stats.gamma.fit(dur)
# create a new random variable using estimated parameters
gamma_rv = stats.gamma.rvs(shape, loc, scale, size=len(dur))

# plot results
plt.figure()
sns.histplot(dur, kde=True, label='MW duration')
sns.histplot(gamma_rv, kde=True, label=f'Gamma RV alpha:{shape:.2f} beta:{1/scale:.2f}')
plt.legend()
plt.title(f'MW Duration (mean:{avg:.2f}, median:{med:.2f}, var:{var:.2f}))')
plt.xlabel('reported mind-wandering duration (s)')

# %% QA on onset and offset
# get all samples whose has unfound onset/offset
mask = df_whole['reported_MW'] & (df_whole['win_start'].isnull() | df_whole['win_end'].isnull())
df = df_whole[mask]

# define the destinated directory
des_dir = f'{path}QualityCheckPlot/ReadingImages/UnfoundOnsetOffset/'

# iterate each row to copy plot to the dir for QA
for _, row in df.iterrows():
    run = row['run']
    page = row['page_num']
    sub = row['sub_id']
    try:
        eye = 'R'
        ori_dir = f'{path}{sub}/plots/whole/{eye}/run-{run}/'
        file_name = f'r{run}_{eye}_page{page}.png'
        ori_path = ori_dir + file_name
        des_path = des_dir + sub + '_' + file_name
        shutil.copy2(ori_path, des_path)
    except:
        eye = 'mono'
        ori_dir = f'{path}{sub}/plots/whole/{eye}/run-{run}/'
        file_name = f'r{run}_{eye}_page{page}.png'
        ori_path = ori_dir + file_name
        des_path = des_dir + sub + '_'  + file_name
        shutil.copy2(ori_path, des_path)


# get all samples whose offset is earlier than onset
mask = df_whole['win_start'] > df_whole['win_end']
df = df_whole[mask]

# define the destinated directory
des_dir = f'{path}QualityCheckPlot/ReadingImages/ReversedOnsetOffset/'

# iterate each row to copy plot to the dir for QA
for _, row in df.iterrows():
    run = row['run']
    page = row['page_num']
    sub = row['sub_id']
    try:
        eye = 'R'
        ori_dir = f'{path}{sub}/plots/whole/{eye}/run-{run}/'
        file_name = f'r{run}_{eye}_page{page}.png'
        ori_path = ori_dir + file_name
        des_path = des_dir + sub + '_' + file_name
        shutil.copy2(ori_path, des_path)
    except:
        eye = 'mono'
        ori_dir = f'{path}{sub}/plots/whole/{eye}/run-{run}/'
        file_name = f'r{run}_{eye}_page{page}.png'
        ori_path = ori_dir + file_name
        des_path = des_dir + sub + '_'  + file_name
        shutil.copy2(ori_path, des_path)
        
        
# %% Group plot
plot_eye_features_group(df_whole, df_same, df_last)   
    

# %% Boxplot
plot_eye_features_box(df_whole, df_same, df_last)


# %% Barplot
plot_eye_features_bar(df_whole, df_same, df_last)


# %% Violin plot
plot_eye_features_violin(df_whole, df_same, df_last)


# %% Stats
print('=============Sample Size================')
print('Whole')
mr_size = np.sum(df_whole['win_type']=='MR')
nr_size = np.sum(df_whole['win_type']=='NR')
print(f'MR: {mr_size}\nNR: {nr_size}')

print('Same')
mr_size = np.sum(df_same['win_type']=='MR')
nr_size = np.sum(df_same['win_type']=='NR')
print(f'MR: {mr_size}\nNR: {nr_size}')

print('Last')
mr_size = np.sum(df_last['win_type']=='MR')
nr_size = np.sum(df_last['win_type']=='NR')
print(f'MR: {mr_size}\nNR: {nr_size}')


# %% Plot individual pupil trace
plot_subject_pupiltrace(path)

#%%

# onset
# get all csv file
file_name = '[mono|R]_r[0-9]_PupilTrace_onset.csv'
files = glob(os.path.join(file_path, file_name))

ls = []
for file in files:
    ls.append(np.loadtxt(file, delimiter=",", dtype=float))
    
arr = np.vstack(ls)

fs = 100
time = np.arange(arr.shape[1])/fs

plt.figure()
avg = np.mean(arr, axis=0)
se = np.std(arr, axis=0) / np.sqrt(arr.shape[0])
plt.plot(time, avg, label='mean')
plt.fill_between(time, avg+2*se, avg-2*se, label='CI', alpha=0.2)
plt.xlabel('time after onset word (s)')
plt.ylabel('pupil change (mm)')
plt.title('Pupil Trace after Onset')
plt.savefig(save_path+'pupiltrace_onset.png')


# offset
# get all csv file
file_name = '[mono|R]_r[0-9]_PupilTrace_offset.csv'
files = glob(os.path.join(file_path, file_name))

ls = []
for file in files:
    ls.append(np.loadtxt(file, delimiter=",", dtype=float))
    
arr = np.vstack(ls)

fs = 100
time = -np.flip(np.arange(arr.shape[1])/fs)

plt.figure()
avg = np.mean(arr, axis=0)
se = np.std(arr, axis=0) / np.sqrt(arr.shape[0])
plt.plot(time, avg, label='mean')
plt.fill_between(time, avg+2*se, avg-2*se, label='CI', alpha=0.2)
plt.xlabel('time before offset word (s)')
plt.ylabel('pupil change (mm)')
plt.title('Pupil Trace after Offset')
plt.savefig(save_path+'pupiltrace_offset.png')


# control
# get all csv file
file_name = '[mono|R]_r[0-9]_PupilTrace_control.csv'
files = glob(os.path.join(file_path, file_name))

ls = []
for file in files:
    ls.append(np.loadtxt(file, delimiter=",", dtype=float))
    
arr = np.vstack(ls)

fs = 100
time = np.arange(arr.shape[1])/fs

plt.figure()
avg = np.mean(arr, axis=0)
se = np.std(arr, axis=0) / np.sqrt(arr.shape[0])
plt.plot(time, avg, label='mean')
plt.fill_between(time, avg+2*se, avg-2*se, label='CI', alpha=0.2)
plt.xlabel('time duration (s)')
plt.ylabel('pupil change (mm)')
plt.title('Pupil Trace Control')
plt.savefig(save_path+'pupiltrace_control.png')



# %% Comprehension question analysis
# get the comprehension question answers and responses
cq_df = create_cq_df(path)

# compute correctness
compute_cq_correctness(cq_df)

# compute attempty rate (answered 'I don't know')
compute_cq_attempt_rate(cq_df)

# extract background info
extract_background_info(cq_df)


# %% Plot reading times
time = {reading_name:[] for reading_name in readings}
# loop through every subject
for folder_path in subfolders:
    folder_path += '/log/'
    file_names = os.listdir(folder_path)
    # loop though every trial
    for file_name in file_names:
        if file_name.startswith('.'):
            continue
        # full path to csv file
        data_file = folder_path + file_name
        # extract PsychoPy log info
        df = extract_info(data_file)
        # article name
        reading = df['reading'][0]
        # pages where subject pressed 'space' to advance
        natural_reading_page = np.where(df['key_response'] == 'space')[0]
        # corresponding reading time
        reading_time = list(df['reading_time'][natural_reading_page])
        
        # add to the dict
        time[reading].append(reading_time)
        

# plot reading times
plt.figure(figsize=(12,8))
for index, reading_name in enumerate(readings):
    plt.subplot(2, 3, index+1)
    
    # histogram
    x = time[reading_name]
    plt.hist(x, bins='fd')
    
    # mean
    plt.axvline(np.mean(x), color='k', linestyle='dashed', linewidth=1)
    min_ylim, max_ylim = plt.ylim()
    plt.text(np.mean(x)*1.1, max_ylim*0.9, 'Mean: {:.2f}'.format(np.mean(x)))
    plt.text(np.mean(x)*1.1, max_ylim*0.8, 'std: {:.2f}'.format(np.std(x)))

    # annotate
    plt.xlabel('page time (s)')
    plt.ylabel('count')
    plt.title(f'{reading_name}')
    
plt.tight_layout()
plt.savefig('QA/reading_times.png')
  

# %% Comprehension question score
comp_question_socre = {reading_name:[] for reading_name in readings}
for folder_path in subfolders:
    beh_file_path = glob(os.path.join(folder_path, '*beh_features*'))
    # test empty
    if beh_file_path:
        beh_file_path = beh_file_path[0]
        beh_df = pd.read_csv(beh_file_path)
        # extract percent correct
        quest_correct_indices = np.where(beh_df['percent_correct'].notna())[0]
        for index in quest_correct_indices:
            reading = beh_df['reading'][index]
            percent_correct = beh_df['percent_correct'][index]
            
            # add to the dict
            comp_question_socre[reading].append(percent_correct)

# plot reading times
plt.figure(figsize=(8, 6))
sns.boxplot(data=pd.DataFrame.from_dict(comp_question_socre))
plt.ylabel('Percent Correctness')
plt.tight_layout()
#plt.savefig('QA/comp_question_scores.png')


# %% Individual comprehension questions
page_comp_question = {reading_name:{i:[] for i in np.arange(10)} for reading_name in readings}
for folder_path in subfolders:
    beh_file_path = glob(os.path.join(folder_path, '*beh_features*'))
    # test empty
    if beh_file_path:
        beh_file_path = beh_file_path[0]
        beh_df = pd.read_csv(beh_file_path)
        # loop through each row
        for i in np.arange(beh_df.shape[0]):
            entry = beh_df.iloc[i]
            # read in information
            page = entry['page']
            reading = entry['reading']
            comp = entry['answered_correct']
            
            # add to the dict
            page_comp_question[reading][page].append(int(comp))
            

df = pd.concat({reading: pd.DataFrame(np.mean(score) for _, score in pages.items()).T for reading, pages in page_comp_question.items()}, 
               axis=0)

df = df.reset_index(level=[1], drop=True)

# plot individual page scores
plt.figure(figsize=(12, 8))

for index, reading in enumerate(df.index):
    # get data
    scores = df.iloc[index]
    x_pos = np.arange(len(scores))
    # define subplot
    plt.subplot(2, 3, index+1)
    # create bars
    plt.bar(x_pos, scores)
    # annotate
    plt.xticks(x_pos, x_pos+1)
    plt.ylabel('Precent Correctness')
    plt.xlabel('Page')
    plt.title(reading)

plt.tight_layout()
plt.savefig('QA/individual_page_correctness.png')







