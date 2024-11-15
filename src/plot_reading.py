#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module file contains all function for plotting

Created 1/28/24 by HS
Modified 1/30/24 by HS - add documentations
"""

import numpy as np
import reading_analysis as ra               # for all extra functions
from matplotlib import pyplot as plt        # plotting data
from matplotlib import animation as animation


# location where the reading images and coordinate files are located
READING_DIR = '../../MindlessReading/Reading' # hardcoded. folder, relative to THIS file location


def plot_save_image(out_dir, page, eye, run, win_type=None):
    '''
    This function plot the eye features on the reading page with fixations 
    highlighted in colors based on time window type and their labels (MR vs. NR)

    Parameters
    ----------
    out_dir : string
        DESCRIPTION. the relative path to save the image
    page : object
        DESCRIPTION. see page.py for more details
    eye : string
        DESCRIPTION. L/R eye for bino recording; otherwise mono
    run : int
        DESCRIPTION. run number
    win_type : string, optional
        DESCRIPTION. three different types of time window, over which eye
        features get extracted and calculated 
        The default is None.

    Returns
    -------
    None. But the plot will be saved at proper location

    '''
    # plot the image
    fig, ax = plt.subplots(1,1, figsize = (18, 18))
    # call func to plot reading texts
    plot_reading_page(page, ax)
    # call func to plot fixations
    plot_fixations(page, ax, win_type)
    # annotate and save
    ax.set_title(f'Page Index {page.page_number} - Type {win_type} - Eye {eye}')
    plt.savefig(out_dir + f'r{run}_{eye}_page{page.page_number}.png')
    plt.close()
    

def plot_reading_page(page, ax):
    '''
    Plots reading page with rect on each word on the input axis. 

    Parameters
    ----------
    page : object
        DESCRIPTION. see page.py for more details
    ax : matplotlib axes object
        DESCRIPTION. the axis for plot

    Returns
    -------
    None. reading image is plotted on the axis.

    '''
    # get the reading page
    image_file = f'{READING_DIR}/{page.image_file_location}'
    im = plt.imread(image_file) # get coresponding page image index

    ax.imshow(im, origin='upper', aspect = 'equal', extent=(258, 1662, 1074.6, 5.4)) # hardcode based on screen / image size
    ax.set_xlim([0,1920]) # fix the y axis to the size of the screen display
    ax.set_ylim([0, 1080]) # fix the y axis to the size of the screen display
    ax.set_ylim(ax.get_ylim()[::-1])

    # rectangles to show the bouding box of each word
    word_color = '#CBE4ED'
    for _, true_word in page.true_words.iterrows():

        # define the bounding box of the word
        word_x_start_img_pix = true_word['center_x'] - (true_word['width'] / 2)
        word_x_end_img_pix = true_word['center_x'] + (true_word['width'] / 2)
        word_y_start_img_pix = true_word['center_y'] - (true_word['height'] / 2)
        word_y_end_img_pix = true_word['center_y'] + (true_word['height'] / 2)

        # Convert image word pixels to eyelink pixels
        word_x_start,word_y_start = ra.convert_pixel_to_eyelink(word_x_start_img_pix, word_y_start_img_pix)
        word_x_end, word_y_end = ra.convert_pixel_to_eyelink(word_x_end_img_pix, word_y_end_img_pix)

        # update color of highlight
        if (true_word['is_error']): # if is_error
            word_color = "#f77959" # red-ish color
            # just plot the clicked word
            ax.add_patch(plt.Rectangle((word_x_start, word_y_start), 
                                       word_x_end-word_x_start, 
                                       word_y_end-word_y_start, 
                                       alpha=0.5, color=word_color))
        else:
            word_color = '#CBE4ED' # light grey



def plot_fixations(page, ax, win_type=None):
    '''
    Plots eye fixations as circles on the input axis. Color represents the
    occurence timing and size represents fixation duration. If fixations fall 
    within valid time window based on win_type, they will be center colored 
    differently for NR and MR. 

    Parameters
    ----------
    page : object
        DESCRIPTION. see page.py for more details
    ax : matplotlib axes object
        DESCRIPTION. the axis for plot
    win_type : string, optional
        DESCRIPTION. three different types of time window, over which eye
        features get extracted and calculated 
        The default is None.

    Returns
    -------
    None. fixations is plotted on the axis. 

    '''
    
    # get the fixation data from page object
    fix_data = page.fixations
    # gaze pos x
    x_data = np.array(fix_data['xAvg'])
    # gaze pos y
    y_data = np.array(fix_data['yAvg'])
    # pupil size
    data_size = np.array(fix_data['duration'])/3
    
    # scatter plot the fixations as dots
    ax.scatter(x_data, y_data, c=np.arange(1,len(x_data)+1), s=data_size, alpha=0.7)
    
    # plot a line between two fixations
    for index in np.arange(1, len(x_data)):
        x = [x_data[index-1], x_data[index]]
        y = [y_data[index-1], y_data[index]]
        ax.plot(x, y, 'r', alpha=0.3)
    
    
    # color fixation center differently based on type of time window
    if win_type in ['whole','last']:
        color = 'r' if page.win_type == -1 else 'b'
        fix_mask = fix_data['in_win']==page.win_type
        # differentiate fixations within the reported time window
        ax.scatter(x_data[fix_mask], y_data[fix_mask], c=color, 
                   s=data_size[fix_mask]/5, alpha=0.9)
    
    # in this case, plot MR in red and NR in green only for page with valid
    # reported mw
    elif win_type == 'same':
        if page.valid_mw:
            fix_mask_MR = fix_data['in_win']==-1
            ax.scatter(x_data[fix_mask_MR], y_data[fix_mask_MR], c='r', 
                       s=data_size[fix_mask_MR]/5, alpha=0.9)
            fix_mask_NR = fix_data['in_win']==1
            ax.scatter(x_data[fix_mask_NR], y_data[fix_mask_NR], c='b', 
                       s=data_size[fix_mask_NR]/5, alpha=0.9)


def plot_normalized_data(pages):
    '''
    Handle function call for the normalized plot
    input:  pages - array of page objects
    output: NONE - show plot
    '''
    fig, ax1 = plt.subplots(1,1, figsize = (18, 18))
    ra.plot_norm_data(pages, ax1)
    plt.show()
    
    

def plot_individual_pages(page_index, pages):
    '''
    Handle function call for the individual plot
    input:  pages - array, page objects
            page_index - int, page to plot
    output: NONE - show plot
    '''

    if page_index == 'all':
        for i in range(len(pages)):
            fig, ax1 = plt.subplots(1,1, figsize = (18, 18))
            page_index = i
            image_files = f'{READING_DIR}/{pages[page_index].image_file_location}'
            ra.no_animate(image_files, pages[i], ax1)
            plt.show()
    else:
        page_index = int(page_index)
        fig, ax1 = plt.subplots(1,1, figsize = (18, 18))
        image_files = f'{READING_DIR}/{pages[page_index].image_file_location}'
        ra.no_animate(image_files, pages[page_index], ax1)
        plt.show()
        

def plot_animation_page(page_index, pages):
    '''
    Handle function call for the animating plot
    input:  pages - array, page objects
            page_index - int, page to plot
    output: NONE - show plot
    '''
    fig, ax1 = plt.subplots(1,1, figsize = (18, 18))
    image_files = f'{READING_DIR}/{pages[page_index].image_file_location}'
    ani = animation.FuncAnimation(fig, ra.animate, interval=20, blit=False, fargs=(ax1, page_index, image_files, pages))
    plt.show()


def no_animate(file_location, page_data, ax, eye=None):
    '''
    input:  file_location - location of the image (from Page object)
            page_data - the page object of interest
            ax - which axis to plot the data on
    output: None, show's plot, close plot to show next image (if available)
    '''
    im = plt.imread(file_location) # get coresponding page image index
    

    ax.imshow(im, origin='upper', aspect = 'equal', extent=(258, 1662, 1074.6, 5.4)) # hardcode based on screen / image size
    ax.set_xlim([0,1920]) # fix the y axis to the size of the screen display
    ax.set_ylim([0, 1080]) # fix the y axis to the size of the screen display
    ax.set_ylim(ax.get_ylim()[::-1])
    
    fix_data = page_data.feature_data['fix']
    # gaze pos x
    x_data = np.array(fix_data['xAvg'])
    # gaze pos y
    y_data = np.array(fix_data['yAvg'])
    # pupil size
    data_size = np.array(fix_data['duration'])/3
    
    # scatter plot the fixations as dots
    ax.scatter(x_data, y_data, c=np.arange(1,len(x_data)+1), s=data_size, alpha=0.7)
    
    # differentiate fixations within the reported time window
    if page_data.mw_reported:
        fix_mask = np.array(fix_data['in_win'], dtype=bool)
        # scatter plot the fixations as dots
        ax.scatter(x_data[fix_mask], y_data[fix_mask], c='r', 
                   s=data_size[fix_mask]/2, alpha=0.9)
    
    # plot a line between two fixations
    for index in np.arange(1, len(x_data)):
        x = [x_data[index-1], x_data[index]]
        y = [y_data[index-1], y_data[index]]
        ax.plot(x, y, 'r', alpha=0.3)
    

    # rectangles to show the bouding box of each word
    word_color = '#CBE4ED'
    for _, true_word in page_data.true_words.iterrows():

        # define the bounding box of the word
        word_x_start_img_pix = true_word['center_x'] - (true_word['width'] / 2)
        word_x_end_img_pix = true_word['center_x'] + (true_word['width'] / 2)
        word_y_start_img_pix = true_word['center_y'] - (true_word['height'] / 2)
        word_y_end_img_pix = true_word['center_y'] + (true_word['height'] / 2)

        # Convert image word pixels to eyelink pixels
        word_x_start,word_y_start = convert_pixel_to_eyelink(word_x_start_img_pix, word_y_start_img_pix)
        word_x_end, word_y_end = convert_pixel_to_eyelink(word_x_end_img_pix, word_y_end_img_pix)

        # update color of highlight
        if (true_word['is_error']): # if is_error
            word_color = "#f77959" # red-ish color
            # just plot the clicked word
            ax.add_patch(plt.Rectangle((word_x_start, word_y_start), word_x_end-word_x_start, word_y_end-word_y_start, alpha=0.5, color=word_color))
        else:
            word_color = '#CBE4ED' # light grey

        #ax.add_patch(plt.Rectangle((word_x_start, word_y_start), word_x_end-word_x_start, word_y_end-word_y_start, alpha=0.2, color=word_color))

    ax.set_title(f'Page Index {page_data.page_number} - Fixations with Duration - {page_data.page_view} - {page_data.error_type} - Eye {eye}')

def plot_norm_data(pages_data, ax):
    '''
    Plot nomalized data for all pages in the series.
    input:  pages_data - array of Page objects
            ax - axis to plot the data
    output  None. Creates plot
    '''

    # initialize array to store the page data
    num_fix_pages = []
    pupil_slopes_pages = []
    num_blinks_pages = []
    num_saccades_pages = []

    # add each page data to the array
    for page in pages_data:
        num_fix_pages.append(page.num_fixations)
        pupil_slopes_pages.append(page.pupil_slope)
        num_blinks_pages.append(page.num_blinks)
        num_saccades_pages.append(page.num_saccades)

    # plot the data
    ax.plot(np.arange(0,len(pages_data)), num_fix_pages, c='r', label='fixations')
    ax.plot(np.arange(0,len(pages_data)), pupil_slopes_pages, c='b', label='pupil slope')
    ax.plot(np.arange(0,len(pages_data)), num_blinks_pages, c='m', label='blinks')
    ax.plot(np.arange(0,len(pages_data)), num_saccades_pages, c='k', label='saccades')

    # format the plot
    ax.set_title("Normalized Data for All Pages")
    ax.legend()


def animate(i, ax, page_index, file_location, pages_data):
    '''
    input: i - the index of the data to show
            pages_data - dictionary of data broken down by pages
            page_index - the page to plot
    output: None, show's plot, close plot to show next image (if available)
    '''
    im = plt.imread(file_location)
    ax.cla() # clear the previous image
    ax.imshow(im, origin='upper', aspect = 'equal', extent=(258, 1662, 1074.6, 5.4))
    

    ax.set_xlim([0,1920]) # fix the y axis
    ax.set_ylim([0, 1080]) # fix the y axis
    ax.set_ylim(ax.get_ylim()[::-1])
    x_data = np.array(pages_data[page_index].feature_data['fix'][0])[:i]
    y_data = np.array(pages_data[page_index].feature_data['fix'][1])[:i]
    # data_size = np.array(pages_data[page_index].feature_data['fix'][2])
    # ax.scatter(pages_data[page_index].feature_data['fix'][0]-x_offset, pages_data[page_index].feature_data['fix'][1]-y_offset) # plot the line
    # ax.scatter(x_data, y_data, c=np.arange(1,len(pages_data[page_index].feature_data['fix'][0])+1), s = data_size) # plot the line
    ax.scatter(x_data, y_data, c=np.arange(1,len(pages_data[page_index].feature_data['fix'][0])+1)[:i]) # plot the line

    # format the plot
    ax.set_title(f'Page Index {page_index} Fixations')

def no_animate(file_location, page_data, ax, eye=None):
    '''
    input:  file_location - location of the image (from Page object)
            page_data - the page object of interest
            ax - which axis to plot the data on
    output: None, show's plot, close plot to show next image (if available)
    '''
    im = plt.imread(file_location) # get coresponding page image index
    

    ax.imshow(im, origin='upper', aspect = 'equal', extent=(258, 1662, 1074.6, 5.4)) # hardcode based on screen / image size
    ax.set_xlim([0,1920]) # fix the y axis to the size of the screen display
    ax.set_ylim([0, 1080]) # fix the y axis to the size of the screen display
    ax.set_ylim(ax.get_ylim()[::-1])
    
    fix_data = page_data.feature_data['fix']
    # gaze pos x
    x_data = np.array(fix_data['xAvg'])
    # gaze pos y
    y_data = np.array(fix_data['yAvg'])
    # pupil size
    data_size = np.array(fix_data['duration'])/3
    
    # scatter plot the fixations as dots
    ax.scatter(x_data, y_data, c=np.arange(1,len(x_data)+1), s=data_size, alpha=0.7)
    
    # differentiate fixations within the reported time window
    if page_data.mw_reported:
        fix_mask = np.array(fix_data['in_win'], dtype=bool)
        # scatter plot the fixations as dots
        ax.scatter(x_data[fix_mask], y_data[fix_mask], c='r', 
                   s=data_size[fix_mask]/2, alpha=0.9)
    
    # plot a line between two fixations
    for index in np.arange(1, len(x_data)):
        x = [x_data[index-1], x_data[index]]
        y = [y_data[index-1], y_data[index]]
        ax.plot(x, y, 'r', alpha=0.3)
    

    # rectangles to show the bouding box of each word
    word_color = '#CBE4ED'
    for _, true_word in page_data.true_words.iterrows():

        # define the bounding box of the word
        word_x_start_img_pix = true_word['center_x'] - (true_word['width'] / 2)
        word_x_end_img_pix = true_word['center_x'] + (true_word['width'] / 2)
        word_y_start_img_pix = true_word['center_y'] - (true_word['height'] / 2)
        word_y_end_img_pix = true_word['center_y'] + (true_word['height'] / 2)

        # Convert image word pixels to eyelink pixels
        word_x_start,word_y_start = convert_pixel_to_eyelink(word_x_start_img_pix, word_y_start_img_pix)
        word_x_end, word_y_end = convert_pixel_to_eyelink(word_x_end_img_pix, word_y_end_img_pix)

        # update color of highlight
        if (true_word['is_error']): # if is_error
            word_color = "#f77959" # red-ish color
            # just plot the clicked word
            ax.add_patch(plt.Rectangle((word_x_start, word_y_start), word_x_end-word_x_start, word_y_end-word_y_start, alpha=0.5, color=word_color))
        else:
            word_color = '#CBE4ED' # light grey

        #ax.add_patch(plt.Rectangle((word_x_start, word_y_start), word_x_end-word_x_start, word_y_end-word_y_start, alpha=0.2, color=word_color))

    ax.set_title(f'Page Index {page_data.page_number} - Fixations with Duration - {page_data.page_view} - {page_data.error_type} - Eye {eye}')

def plot_norm_data(pages_data, ax):
    '''
    Plot nomalized data for all pages in the series.
    input:  pages_data - array of Page objects
            ax - axis to plot the data
    output  None. Creates plot
    '''

    # initialize array to store the page data
    num_fix_pages = []
    pupil_slopes_pages = []
    num_blinks_pages = []
    num_saccades_pages = []

    # add each page data to the array
    for page in pages_data:
        num_fix_pages.append(page.num_fixations)
        pupil_slopes_pages.append(page.pupil_slope)
        num_blinks_pages.append(page.num_blinks)
        num_saccades_pages.append(page.num_saccades)

    # plot the data
    ax.plot(np.arange(0,len(pages_data)), num_fix_pages, c='r', label='fixations')
    ax.plot(np.arange(0,len(pages_data)), pupil_slopes_pages, c='b', label='pupil slope')
    ax.plot(np.arange(0,len(pages_data)), num_blinks_pages, c='m', label='blinks')
    ax.plot(np.arange(0,len(pages_data)), num_saccades_pages, c='k', label='saccades')

    # format the plot
    ax.set_title("Normalized Data for All Pages")
    ax.legend()


def animate(i, ax, page_index, file_location, pages_data):
    '''
    input: i - the index of the data to show
            pages_data - dictionary of data broken down by pages
            page_index - the page to plot
    output: None, show's plot, close plot to show next image (if available)
    '''
    im = plt.imread(file_location)
    ax.cla() # clear the previous image
    ax.imshow(im, origin='upper', aspect = 'equal', extent=(258, 1662, 1074.6, 5.4))
    

    ax.set_xlim([0,1920]) # fix the y axis
    ax.set_ylim([0, 1080]) # fix the y axis
    ax.set_ylim(ax.get_ylim()[::-1])
    x_data = np.array(pages_data[page_index].feature_data['fix'][0])[:i]
    y_data = np.array(pages_data[page_index].feature_data['fix'][1])[:i]
    # data_size = np.array(pages_data[page_index].feature_data['fix'][2])
    # ax.scatter(pages_data[page_index].feature_data['fix'][0]-x_offset, pages_data[page_index].feature_data['fix'][1]-y_offset) # plot the line
    # ax.scatter(x_data, y_data, c=np.arange(1,len(pages_data[page_index].feature_data['fix'][0])+1), s = data_size) # plot the line
    ax.scatter(x_data, y_data, c=np.arange(1,len(pages_data[page_index].feature_data['fix'][0])+1)[:i]) # plot the line

    # format the plot
    ax.set_title(f'Page Index {page_index} Fixations')
    