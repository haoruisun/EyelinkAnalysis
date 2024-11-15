# -*- coding: utf-8 -*-
"""
This module contains functions that match clicks to actual word from the reading text. 

Created on 10/29/24 by HS

"""
import pandas as pd
import numpy as np
from .utils import convert_height_unit_to_pixel

def find_match(dfWords, coord_info, dist_max=1000):
    '''
    Match the clicks to words. 

    Parameters
    ----------
    dfWords : DataFrame
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
    words_x_start = dfWords['center_x'] - dfWords['width']/2
    words_x_end = dfWords['center_x'] + dfWords['width']/2
    words_y_start = dfWords['center_y'] - dfWords['height']/2
    words_y_end = dfWords['center_y'] + dfWords['height']/2
    
    # get x and y for the click
    pos_x, pos_y = coord_info
    
    # compute the distance between click and word boundry box
    dist_x_left = (words_x_start - pos_x)
    dist_x_right = (pos_x - words_x_end)
    dist_y_top = (words_y_start - pos_y)
    dist_y_bottom = (pos_y - words_y_end)
    
    # find the maximum distance from click to the word for x and y
    max_x = np.max(np.vstack((dist_x_left, dist_x_right, np.zeros(len(dist_x_left)))), axis=0)
    max_y = np.max(np.vstack((dist_y_top, dist_y_bottom, np.zeros(len(dist_y_top)))), axis=0)
    
    # calculate the distance using x and y
    dist = np.sqrt(np.square(max_x) + np.square(max_y))
    
    # check if the minimum dist exceeds threshold values
    if np.min(dist) < dist_max:
        matched_index = np.argmin(dist)
    else:
        matched_index = -1
    
    # return the index that has the shortest distance
    return matched_index


def match_clicks2words(data_file, reading_path = '../../MindlessReading/Reading'):
    '''
    Read PsychoPy csv file and match any click to words.

    Parameters
    ----------
    data_file : string
        DESCRIPTION. The path to PsychoPy csv file.
    reading_path: string
        DESCRIPTION. The prefix path to reading materials.
        Default: '../../MindlessReading/Reading'

    Returns
    -------
    matched_words : list of list (tuple)
        DESCRIPTION. len(matched_words) matches number of reported MW in csv file
        For each report, two words matched to two clicks are returned. 

    '''
    # load task csv file
    df = pd.read_csv(data_file)
    
    # create empty list to store words matched to clicks
    matched_words = pd.DataFrame({'page': np.arange(10), 
                                  'first_word': np.full(10, None),
                                  'first_index': np.full(10, np.nan),
                                  'second_word': np.full(10, None),
                                  'second_index': np.full(10, np.nan)})
    
    # if na clicks logged in the PsychoPy .csv file, return empty dataframe
    if 'first_click.x' not in df.columns:
        return matched_words
    
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

    # loop through each page to match clicks to words
    for index, page_num in page.items():
        # get the words for current page
        coord_page_df = coord_df[coord_df['page'] == page_num]
        
        # loop through two clicks
        for click_pos, order in zip([first_click_pos, second_click_pos], ['first', 'second']):
            # get the x and y coords 
            x_py, y_py = click_pos.loc[index]
            # convert from height unit to pixel unit
            x_pixel, y_pixel = convert_height_unit_to_pixel(x_py, y_py)
            
            # call function to get the index for matched word to the first click
            matched_index = find_match(coord_page_df, (x_pixel, y_pixel))
            word = coord_page_df['words'].iloc[matched_index]
            # store matched word to the dataframe
            matched_words.loc[matched_words.page==page_num, f'{order}_word'] = word
            matched_words.loc[matched_words.page==page_num, f'{order}_index'] = matched_index
    
    return matched_words





