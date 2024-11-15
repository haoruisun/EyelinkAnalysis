# -*- coding: utf-8 -*-
"""
This module contains functions that parses information from the PsychoPy results .csv file.  

Created on 10/29/24 by HS

"""
import math
import numpy as np
import pandas as pd
from .utils import convert_error_page_pixel_to_py

def extract_cq_info(data_file):
    '''
    This function reads in psychopy csv file and extracts answers and subject
    responses to the comprehension questions

    Parameters
    ----------
    data_file : string
        DESCRIPTION. The path to PsychoPy csv file.

    Returns
    -------
    answer df: dataframe
        DESCRIPTION. df contains answers and subject responses

    '''
    # read the csv file
    df = pd.read_csv(data_file)
    # locate to rows that store answers and responses
    mask = np.where(pd.isnull(df['key_resp_question.keys']) == False)[0]
    reading = df.loc[mask, 'reading']
    answers = df.loc[mask, 'Answer']
    responses = df.loc[mask, 'key_resp_question.keys']
    # create the return dataframe with these two series
    frame = {'reading': reading, 'answers': answers, 'responses':responses}
    
    return pd.DataFrame(frame)


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
    # define local variables
    reading_path = '../../MindlessReading/Reading'

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