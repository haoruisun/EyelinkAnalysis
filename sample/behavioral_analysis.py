# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 09:10:53 2022

@author: nickm, haoruis

This script extract the behavioral data from PsychoPy .csv files, and generate
individual and group machine learning dataset. 

"""
# %% Packages
import pandas as pd
import numpy as np
import seaborn as sns
import math
import re
import os
from matplotlib import pyplot as plt
from glob import glob
from sklearn.decomposition import PCA


# %% Constants
# define root path relative to current working directory
root_path = '../../../DataCollection/'
reading_path = '../../MindlessReading/Reading'
ml_path = 'MachineLearningData/'
subfolders = [f.path for f in os.scandir(root_path) if f.is_dir()]
slider_col = ['slider_task', 'slider_detailed', 'slider_words', 'slider_emotion']
ml_col = ['num_blinks', 'num_fixations', 'norm_blink_freq', 
          'pupil_slope', 'mean_pupil_size', 'norm_num_word_fix', 
          'norm_in_word_regression', 'norm_out_word_regression', 
          'zipf_duration_correlation', 'label', 'sub_id']

features = ['num_blinks', 'num_fixations', 'pupil_slope', 'mean_pupil_size', 
            'norm_in_word_regression', 'norm_out_word_regression', 
            'zipf_duration_correlation']

# %% Helper functions
def extract_info(data_file):
    '''
    This function extracts information from the PsychoPy log file and returns
    them in a dataframe.
    
    This function runs for each reading
​
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
​
    Parameters
    ----------
    data_file : string
        DESCRIPTION. The absolute path of PsychoPy log file (.csv)
​
    Returns
    -------
    DataFrame
        DESCRIPTION. The dataframe that contians desired information from the
                     log file.
​
    '''
    #Load task csv file
    df = pd.read_csv(data_file)
    #Compute the valid indices based on the error type
    error_type = df['error_type']
    valid_indices = np.where(pd.isnull(error_type) == False)[0]
    
    #Extract the desired information and store them in Series
    error_type = error_type[valid_indices].reset_index(drop=True)               #Type of page array
    key_resp = df['key_resp.keys'][valid_indices].reset_index(drop=True)        #Button pressed array
    reading_time = df['key_resp.rt'][valid_indices].reset_index(drop=True)
    
    #Create array with coordinates of subject's mouse clicks (nan when no click on a page)
    try:
        mouse_error = pd.Series([item for item in zip(df['mouse_error.x'], df['mouse_error.y'])])\
            [valid_indices].reset_index(drop=True)
    except KeyError: #When no clicks are made in a reading, create array (nan,nan)
        fill = (np.nan,np.nan)
        mouse_error = pd.Series([fill for item in error_type])
        
    #Slider values
    slider_task = df['slider_task.response'][valid_indices].reset_index(drop=True)
    slider_detailed = df['slider_detailed.response'][valid_indices].reset_index(drop=True)
    slider_words = df['slider_words.response'][valid_indices].reset_index(drop=True)
    slider_emotion = df['slider_emotion.response'][valid_indices].reset_index(drop=True)

    #Page and reading time
    page = df['page'][valid_indices].reset_index(drop=True)
    reading = df['reading'][valid_indices].reset_index(drop=True)

    #Construct paths to error files
    current_reading = reading[0].lower().replace(' ', '_')
    lex_coord_path = f'{reading_path}/{current_reading}/{current_reading}_lexical/{current_reading}_lexical_coordinates.csv'
    gib_coord_path = f'{reading_path}/{current_reading}/{current_reading}_gibberish/{current_reading}_gibberish_coordinates.csv'
    lex_coordfile = pd.read_csv(lex_coord_path)
    gib_coordfile = pd.read_csv(gib_coord_path)
    
    #Aquire comprehension questions
    questions_path = f'ComprehensionQuestions/{current_reading}_questions.xlsx'
    questions_file = pd.read_excel(questions_path)
    
    #Get the answers to the comp. questions and the subject's answers
    correct_answers = [x for x in questions_file['Answer'].to_numpy() if str(x) != 'nan']
    subject_answers = [x for x in df['key_resp_question.keys'].to_numpy() if str(x) != 'nan']
    are_answers_correct = [True if i ==j else False for i,j in zip(subject_answers[0:-2],correct_answers)]
    
    #Initialize arrays
    is_detected = []
    error_words = []
    
    #For each page in the reading
    for index in range(len(page)):
        
        current_page = page[index]
        current_error_type = error_type[index]
        
        #Look at the right coordinate file
        if (current_error_type != 'control'):
            if (current_error_type == 'lexical'):
                file = lex_coordfile
            elif (current_error_type == 'gibberish'):
                file = gib_coordfile

            #Get the necessary info from the coordinate file
            is_error = file['is_error'].to_numpy()      #Boolean array -> is each word in the reading an error
            words = file['words'].to_numpy()            #All words in the reading
            center_x = file['center_x'].to_numpy()      #X-coord. of each word (in pixels)
            center_y = file['center_y'].to_numpy()      #Y-coord. of each word (in pixels)
            pagenums = file['page'].to_numpy()          #The page number each word is on
            width = file['prop_width'].to_numpy()       #Width of each word
            height = file['prop_height'].to_numpy()     #height of each word
            
            #Find the error words on the current page
            error_words.append(words[((is_error == 1) & (pagenums == current_page))])
            
            #Find coordinates and width/height of words on the current page
            xpos = center_x[((is_error == 1) & (pagenums == current_page))]
            ypos = center_y[((is_error == 1) & (pagenums == current_page))]
            word_width = width[((is_error == 1) & (pagenums == current_page))]
            word_height = height[((is_error == 1) & (pagenums == current_page))]
            
            #Convert coordinates from pixels to task format
            xpos,ypos = convert_error_page_pixel_to_py(xpos,ypos)
            
            #Determine if the subject clicked
            if (math.isnan(mouse_error[index][0])):
                is_detected.append(False)
            #If the subject clicked, test to see if the click was on an error word
            #(Within a threshold of the x and y position of the error word)
            elif (((xpos - word_width) < mouse_error[index][0]) & (mouse_error[index][0] < (xpos + word_width)) & ((ypos - word_height) < mouse_error[index][1]) & (mouse_error[index][1] < (ypos + word_height))).any():
                    is_detected.append(True)
                    
            #If the subject clicked incorrectly
            else:
                is_detected.append(False)
        else:
            is_detected.append(None)
            error_words.append(None)
     
    #Create arrays filled with None values
    percent_detected = [None] * len(page)
    percent_correct = [None] * len(page)
    #Put the percent of detected words for a reading in the first index
    percent_detected[0] = np.average([i for i in is_detected if i is not None])
    #Put the percent of correct comprehension answeres for a reading in the first index
    percent_correct[0] = np.average(are_answers_correct)
    

    # generate the frame with extracted Series
    frame = {'page': page, 'reading': reading, 'reading_time': reading_time,
             'error': error_type, 'key_response': key_resp,
             'mouse_error': mouse_error, 'slider_task': slider_task,
             'slider_detailed': slider_detailed, 'slider_words': slider_words,
             'slider_emotion': slider_emotion,'error_words': error_words,
             'is_detected': is_detected, 'answered_correct': are_answers_correct,
             'percent_detected': percent_detected, 'percent_correct': percent_correct}

    return pd.DataFrame(frame)


def convert_error_page_pixel_to_py(x_image, y_image):
    '''
    Convert error page pixel to psychopy height unit

    Parameters
    ----------
    x_image : float
        DESCRIPTION. position x in pixel
    y_image : float
        DESCRIPTION. position y in pixel

    Returns
    -------
    x_py : float
        DESCRIPTION. position x in height unit
    y_py : float
        DESCRIPTION. position y in height unit

    '''
    x_py = (x_image-1900/2)/1900 * 1.12
    y_py = (-y_image+1442/2)/1442 * 0.85 + 0.05
    
    return x_py, y_py


def convert_height_unit_to_pixel(x_height, y_height):
    '''
    Convert height unit to pixel for image (1900 x 1442 pixels) displayed at 
    pos (0, 0) w/ size (1.3, 0.99) in PsychoPy.

    Parameters
    ----------
    x_height : float
        DESCRIPTION. PsychoPy height unit
    y_height : float
        DESCRIPTION. PsychoPy height unit

    Returns
    -------
    x_pixel : float
        DESCRIPTION. Image pixel unit
    y_pixel : float 
        DESCRIPTION. Image pixel unit 

    '''
    x_pixel = x_height/1.3*1900 + 1900/2
    y_pixel = -y_height/0.99*1442 + 1442/2
    return x_pixel, y_pixel


def find_match(words_info, click_pos):
    '''
    Match the clicks to words. 

    Parameters
    ----------
    words_info : DataFrame
        DESCRIPTION. The dataframe for a single page. It should at least contain
        columns 'center_x', 'center_y', 'width', and 'height', which are 
        coordinate information of words in pixel unit
    click_pos : Tuple
        DESCRIPTION. The coordinate info (x, y) for a click in pixel unit

    Returns
    -------
    matched_index: int
        DESCRIPTION. The index value of matched word for input dataframe. 

    '''
    # compute the start and end positions of words
    words_x_start = words_info['center_x'] - words_info['width']/2
    words_x_end = words_info['center_x'] + words_info['width']/2
    words_y_start = words_info['center_y'] - words_info['height']/2
    words_y_end = words_info['center_y'] + words_info['height']/2
    
    # get x and y for the click
    click_x, click_y = click_pos
    
    # compute the distance between click and word boundry box
    dist_x_left = (words_x_start - click_x)
    dist_x_right = (click_x - words_x_end)
    dist_y_top = (words_y_start - click_y)
    dist_y_bottom = (click_y - words_y_end)
    
    # find the maximum distance from click to the word for x and y
    max_x = np.max(np.vstack((dist_x_left, dist_x_right, np.zeros(len(dist_x_left)))), axis=0)
    max_y = np.max(np.vstack((dist_y_top, dist_y_bottom, np.zeros(len(dist_y_top)))), axis=0)
    
    # calculate the distance using x and y
    dist = np.sqrt(np.square(max_x) + np.square(max_y))
    
    # return the index that has the shortest distance
    return np.argmin(dist)


def match_clicks_to_words(data_file):
    '''
    Read PsychoPy csv file and match any click to words.

    Parameters
    ----------
    data_file : string
        DESCRIPTION. The path to PsychoPy csv file.

    Returns
    -------
    matched_words : list of list (tuple)
        DESCRIPTION. len(matched_words) matches number of reported MW in csv file
        For each report, two words matched to two clicks are returned. 

    '''
    # load task csv file
    df = pd.read_csv(data_file)
    
    # keep rows with valid clicks
    valid_indices = np.where(pd.isnull(df['first_click.x']) == False)[0]
    df = df.iloc[valid_indices]
    
    # get the positions of clicks
    # first click
    first_click_pos = df[['first_click.x', 'first_click.y']]
    # second click
    second_click_pos = df[['second_click.x','second_click.y']]
    
    # get the associated reading name and page
    reading = df['reading'].iloc[0].lower().replace(' ', '_')
    page = df['page']
    
    # load the coordinate file
    coord_path = f'{reading_path}/{reading}/{reading}_control/{reading}_control_coordinates.csv'
    coord_df = pd.read_csv(coord_path)
    
    # create empty list to store words matched to clicks
    matched_words = pd.DataFrame({'page': np.arange(10), 
                                  'first_word': np.full(10, np.nan),
                                  'first_index': np.full(10, np.nan),
                                  'seccond_word': np.full(10, np.nan),
                                  'second_index': np.full(10, np.nan)})
    # loop through each page to match clicks to words
    for page_num in page:
        # get the words for current page
        coord_page_df = coord_df[coord_df['page'] == page_num]
        
        # loop through two clicks
        for click_pos, order in zip([first_click_pos, second_click_pos], ['first', 'second']):
            # get the x and y coords 
            x_py, y_py = click_pos.iloc[index]
            # convert from height unit to pixel unit
            x_pixel, y_pixel = convert_height_unit_to_pixel(x_py, y_py)
            
            # call function to get the index for matched word to the first click
            matched_index = find_match(coord_page_df, (x_pixel, y_pixel))
            word = coord_page_df['words'].iloc[matched_index]
            # store matched word to the dataframe
            matched_words[f'{order}_word'].iloc[page_num] = word
            matched_words[f'{order}_index'].iloc[page_num] = matched_index
    
    return matched_words


# %% Extract behavior info
for folder_path in subfolders:
    # extract subject id
    subject_id = re.findall(r's[0-9]+', folder_path)[0]
    #Find the csv files for each reading
    log_files = os.listdir(f'{folder_path}/log')
    
    subject_labels = pd.DataFrame()
    
    for log_file in log_files:
        if log_file.endswith('.csv'):
            #Extract features for a reading
            dfr = extract_info(f'{folder_path}/log/{log_file}')
            #Append reading info to rest of subject trial data
            subject_labels = pd.concat([subject_labels,dfr]) 
        
    
    #Write subject features out to csv
    subject_labels.to_csv(f'{folder_path}/{subject_id}_beh_features.csv')
  
    
# %% Extract slider values
def extract_slider_value(file_path):
    '''
    This function extracts slider values from the behavior csv file

    Parameters
    ----------
    file_path : string
        DESCRIPTION. The path to anaylzed behavior file

    Returns
    -------
    df_slider : dataframe (n x 4)
        DESCRIPTION. The dataframe that includes all slider responese

    '''
    # load behavior csv file
    df_beh = pd.read_csv(file_path, index_col=0)
    
    # define empty dataframe to store slider values
    slider_col = ['slider_task', 'slider_detailed', 'slider_words', 'slider_emotion']
    df_slider = pd.DataFrame(columns=slider_col)
    
    # extract slider values
    df = df_beh.loc[:, slider_col].dropna()  
    df_slider = pd.concat([df_slider, df], ignore_index=True)
    
    return df_slider 
    


# %% Generate subject dataset
for folder_path in subfolders:
    # extract subject id
    subject_id = re.findall(r's[0-9]+', folder_path)[0]
    # extract tracking and behavior features
    eye_file_path = glob(os.path.join(folder_path, '*[mono|R]_features*'))
    beh_file_path = glob(os.path.join(folder_path, '*beh_features*'))
    # if they are both there
    if eye_file_path and beh_file_path:
        df_eye = pd.read_csv(eye_file_path[0], index_col=0)
        df_eye.rename(columns={'page_num':'page'}, inplace=True)
        df_beh = pd.read_csv(beh_file_path[0], index_col=0)
        df_ml = pd.merge(df_eye, df_beh, on=['reading', 'page'], how='inner')
        
        # =========== use PCA to extract mind wandering features ==============
        df_slider = df_ml.loc[:, slider_col].dropna()
        
        # apply PCA on the slider values
        pca = PCA().fit(df_slider)
        pca_slider = pca.transform(df_slider)
        # compute loading matrix
        # Ref: https://scentellegher.github.io/machine-learning/2020/01/27/pca-loadings-sklearn.html
        loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
        loading_matrix = pd.DataFrame(loadings, columns=['PC1', 'PC2', 'PC3', 'PC3'], 
                                      index=df_slider.columns)
        # get the component that has the highest variance with slider_task
        col_index = np.argmax(np.abs(loading_matrix.loc['slider_task']))
        pca_comp = pca_slider[:, col_index]
        
        # get the sign of correlation
        sign = np.sign(loading_matrix.loc['slider_task'][col_index])
        
        # calculate 1/3 and 2/3 percentiles of selected component
        one_third_percentile = np.percentile(pca_comp, 1/3*100)
        two_third_percentile = np.percentile(pca_comp, 2/3*100)
        
        # convert slider task responses into labels
        # 1: no mind wandering
        # 0: mild mind wandering
        # -1: deep mind wandering
        mw_label = np.zeros(pca_comp.shape)
        mw_label[pca_comp >= two_third_percentile] = 1
        mw_label[pca_comp <= one_third_percentile] = -1
        mw_label *= sign
        
        # store the label in the dataframe
        df_ml['label'] = np.nan
        slider_indices = np.where(pd.isnull(df_ml['slider_task']) == False)[0]
        df_ml.loc[slider_indices, 'label'] = mw_label
        
        # save combined datafile
        df_ml.to_csv(f'{folder_path}/{subject_id}_ML_dataset.csv')
        df_ml.to_csv(f'MachineLearningData/{subject_id}_ML_dataset.csv')
        

# %% Generate group dataset
# combine individual machine learning dataset and extract labels by running 
# PCA on group slider values

# declare empty dataframe to store group dataset
df_ml = pd.DataFrame()
# read in individual ML dataset
ml_file_path = glob(os.path.join(ml_path, 's[0-9]*dataset.csv'))
for file_path in ml_file_path:
    df_sub_ml = pd.read_csv(file_path, index_col=0)
    # extract subject id
    subject_id = re.findall(r's[0-9]+', file_path)[0]
    # add subject id to dataframe
    df_sub_ml['sub_id'] = subject_id
    # concatenate subject dataset to group
    df_ml = pd.concat([df_ml, df_sub_ml], ignore_index=True)


# =========== use PCA to extract mind wandering features ==============
df_slider = df_ml.loc[:, slider_col].dropna()

# apply PCA on the slider values
pca = PCA().fit(df_slider)
pca_slider = pca.transform(df_slider)
# compute loading matrix
# Ref: https://scentellegher.github.io/machine-learning/2020/01/27/pca-loadings-sklearn.html
loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
loading_matrix = pd.DataFrame(loadings, columns=['PC1', 'PC2', 'PC3', 'PC3'], 
                              index=df_slider.columns)
# get the component that has the highest variance with slider_task
col_index = np.argmax(np.abs(loading_matrix.loc['slider_task']))
pca_comp = pca_slider[:, col_index]

# get the sign of correlation
sign = np.sign(loading_matrix.loc['slider_task'][col_index])



# ===================== Ternary Class ========================
# 1: no mind wandering
# 0: mild mind wandering
# -1: deep mind wandering

# calculate 1/3 and 2/3 percentiles of selected component
one_third_percentile = np.percentile(pca_comp, 1/3*100)
two_third_percentile = np.percentile(pca_comp, 2/3*100)

# convert slider task responses into labels
mw_label = np.zeros(pca_comp.shape)
mw_label[pca_comp >= two_third_percentile] = 1
mw_label[pca_comp <= one_third_percentile] = -1
mw_label *= sign

# store the label in the dataframe
slider_indices = np.where(pd.isnull(df_ml['slider_task']) == False)[0]
df_ml.loc[slider_indices, 'label'] = mw_label

# save dataset
df_ml[ml_col].to_csv(f'{ml_path}Group_ML_dataset_ternary.csv')



# ===================== Binary Class (slider values) ========================
# 1: no mind wandering
# -1: mind wandering

# calculate 1/3 and 2/3 percentiles of selected component
middle_percentile = np.percentile(pca_comp, 1/2*100)

# convert slider task responses into labels
mw_label = np.ones(pca_comp.shape)
mw_label[pca_comp <= middle_percentile] = -1
mw_label *= sign

# store the label in the dataframe
slider_indices = np.where(pd.isnull(df_ml['slider_task']) == False)[0]
df_ml.loc[slider_indices, 'label'] = mw_label

# save dataset
df_ml[ml_col].to_csv(f'{ml_path}Group_ML_dataset_binary_slider.csv')



# =================== Binary Class (comprehension questions) ================
# 1: no mind wandering
# 0: mind wandering

# store the label in the dataframe
mw_label = df_ml['answered_correct'].to_numpy(dtype=int)
df_ml['label'] = mw_label

# save dataset
df_ml[ml_col].to_csv(f'{ml_path}Group_ML_dataset_binary_compques.csv')



# %% Generate group dataset (continuous labels)
# combine individual machine learning dataset and extract labels by running 
# PCA on group slider values

# declare empty dataframe to store group dataset
df_ml = pd.DataFrame()
# read in individual ML dataset
ml_file_path = glob(os.path.join(ml_path, 's[0-9]*dataset.csv'))
for file_path in ml_file_path:
    df_sub_ml = pd.read_csv(file_path, index_col=0)
    # extract subject id
    subject_id = re.findall(r's[0-9]+', file_path)[0]
    # add subject id to dataframe
    df_sub_ml['sub_id'] = subject_id
    # concatenate subject dataset to group
    df_ml = pd.concat([df_ml, df_sub_ml], ignore_index=True)


# =========== use PCA to extract mind wandering features ==============
df_slider = df_ml.loc[:, slider_col].dropna()

# apply PCA on the slider values
pca = PCA().fit(df_slider)
pca_slider = pca.transform(df_slider)
# compute loading matrix
# Ref: https://scentellegher.github.io/machine-learning/2020/01/27/pca-loadings-sklearn.html
loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
loading_matrix = pd.DataFrame(loadings, columns=['PC1', 'PC2', 'PC3', 'PC3'], 
                              index=df_slider.columns)
# get the component that has the highest variance with slider_task
col_index = np.argmax(np.abs(loading_matrix.loc['slider_task']))
pca_comp = pca_slider[:, col_index]
    
# plot the distribution of continuous labels
df_slider['pca_loading'] = pca_comp
plt.figure()
sns.histplot(data=df_slider, x='pca_loading', binwidth=0.05, label='pca loading')
sns.histplot(data=df_slider, x='slider_task', binwidth=0.05, label='slider task')
plt.legend()
plt.savefig('QA/pca_loading_distribution.png')

plt.figure()
ax = sns.histplot(data=df_slider, binwidth=0.05, element='step', stat="density")
sns.move_legend(ax, 'upper left')
plt.savefig('QA/slider_responses_distribution.png')

# store the label in the dataframe
slider_indices = np.where(pd.isnull(df_ml['slider_task']) == False)[0]
df_ml.loc[slider_indices, 'label'] = pca_comp

# save dataset
df_ml[ml_col].to_csv(f'{ml_path}Group_ML_dataset_continuous.csv')



# %% Generate group dataset for QA
tau = 20
# declare empty dataframe to store group dataset
df_ml = pd.DataFrame()
# read in individual ML dataset
ml_file_path = glob(os.path.join(ml_path, 's[0-9]*dataset.csv'))
for file_path in ml_file_path:
    df_sub_ml = pd.read_csv(file_path, index_col=0)
    # extract subject id
    subject_id = re.findall(r's[0-9]+', file_path)[0]
    # add subject id to dataframe
    df_sub_ml['sub_id'] = subject_id
    # concatenate subject dataset to group
    df_ml = pd.concat([df_ml, df_sub_ml], ignore_index=True)

# save dataset
df_ml.to_csv(f'QA/Group_dataset_{tau}s.csv')

# plot
fig, axes = plt.subplots(3, 4, figsize=[12, 16], sharex=True)
index = 0
for feature, ax in zip(features, axes.flatten()):
    ax.scatter(df_ml['slider_task'], df_ml[feature])
    ax.set_xlabel('slider task')
    ax.set_title(feature)
plt.tight_layout()
plt.savefig(f'QA/features_vs_slider-task_{tau}s.png')


