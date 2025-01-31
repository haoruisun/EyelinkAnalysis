#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module file contains all function for plotting

Created 1/28/24 by HS
Modified 1/30/24 by HS - add documentations
Modified 1/28/25 by HS - update functions for new pipeline
"""
import os
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt        # plotting data
from matplotlib import animation as animation
from .utils import convert_pixel_to_eyelink


# location where the reading images and coordinate files are located
#READING_DIR = '../../MindlessReading/Reading' # hardcoded. folder, relative to THIS file location


def plot_reading_results(pages, sub_folder, win_type):
    """
    Plots eye fixation data on reading pages, saving the plots with proper filenames.
    
    Parameters
    ----------
    pages : list
        A list of page objects to be processed.
    sub_folder : str
        The base folder path where the plot will be saved.
    win_type : str
        The type of time window for extracting eye features (e.g., 'default', 'same-dur').

    Returns
    -------
    None
        Plots are saved to disk.
    """
    # make folders
    for run_num in range(1,6):
        for eye in ['L', 'R']:
            out_dir = f'{sub_folder}/plot/{win_type}/run{run_num}/{eye}'
            os.makedirs(out_dir, exist_ok=True)

    for page in tqdm(pages, desc="Plotting Reading Pages"):
        for eye in ['L', 'R']:
            # plot the image
            fig, ax = plt.subplots(1,1, figsize = (18, 18))
            # call func to plot reading texts
            plot_reading_page(page, ax)
            # call func to plot fixations
            plot_fixations(page, ax, eye)

            # annotate and save
            ax.set_title(f'Page Index {page.page_number} - Eye {eye}')

            # Define the output directory and file name
            out_dir = f'{sub_folder}/plot/{win_type}/run{page.run_number}/{eye}/'
            filename = f'run{page.run_number}_{eye}_page{page.page_number}.png'
            
            # Check if file already exists and modify the filename if necessary
            counter = 1
            while os.path.exists(out_dir + filename):
                filename = f'run{page.run_number}_{eye}_page{page.page_number}_{counter}.png'
                counter += 1
            
            # Save the figure
            plt.savefig(out_dir + filename)
            plt.close()
    

def plot_reading_page_old(page, ax):
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
    image_folder = f'{os.sep}res{os.sep}pages'
    image_name = page.image_file_location.split(os.sep)[-1]
    image_path = f'{image_folder}{os.sep}{image_name}'

    if os.path.exists(image_path):
        im.plt.imread(image_path)
        ax.imshow(im)
    else:
        # get the reading page
        READING_DIR = f'..{os.sep}..{os.sep}MindlessReading{os.sep}Reading' # hardcoded. folder, relative to THIS file location
        image_path = f'{READING_DIR}{os.sep}{page.image_file_location}'
        im = plt.imread(image_path) # get coresponding page image index

        ax.imshow(im, origin='upper', aspect = 'equal', extent=(258, 1662, 1074.6, 5.4)) # hardcode based on screen / image size
        ax.set_xlim([0,1920]) # fix the y axis to the size of the screen display
        ax.set_ylim([0, 1080]) # fix the y axis to the size of the screen display
        ax.set_ylim(ax.get_ylim()[::-1])

        # rectangles to show the bouding box of each word
        word_color = '#CBE4ED'
        for _, dfWords in page.dfWords.iterrows():

            # define the bounding box of the word
            word_x_start_img_pix = dfWords['center_x'] - (dfWords['width'] / 2)
            word_x_end_img_pix = dfWords['center_x'] + (dfWords['width'] / 2)
            word_y_start_img_pix = dfWords['center_y'] - (dfWords['height'] / 2)
            word_y_end_img_pix = dfWords['center_y'] + (dfWords['height'] / 2)

            # Convert image word pixels to eyelink pixels
            word_x_start,word_y_start = convert_pixel_to_eyelink(word_x_start_img_pix, word_y_start_img_pix)
            word_x_end, word_y_end = convert_pixel_to_eyelink(word_x_end_img_pix, word_y_end_img_pix)
            # update color of highlight
            if (dfWords['is_clicked']): # if it got clicked
                word_color = "#f77959" # red-ish color
                # just plot the clicked word
                ax.add_patch(plt.Rectangle((word_x_start, word_y_start), 
                                        word_x_end-word_x_start, 
                                        word_y_end-word_y_start, 
                                        alpha=0.5, color=word_color))
        
def plot_reading_page(page, ax):
    '''
    This function plots the reading page image and highlights the words that have been clicked (reported) by the user. 
    The words are displayed as colored bounding boxes over the image.

    Args:
        page (object): The page object containing information about the page, such as the image file location and word data.
        ax (matplotlib.axes.Axes): The matplotlib axis object where the page image and highlighted words will be plotted.

    Returns:
        None: The function modifies the `ax` object in place to display the page image and highlighted words.
    '''
    # get the reading page
    READING_DIR = f'..{os.sep}..{os.sep}MindlessReading{os.sep}Reading' # hardcoded. folder, relative to THIS file location
    image_path = f'{READING_DIR}{os.sep}{page.image_file_location}'
    im = plt.imread(image_path) # get coresponding page image index
    ax.imshow(im, origin='upper', aspect = 'equal', extent=(258, 1662, 1074.6, 5.4)) # hardcode based on screen / image size
    ax.set_xlim([0,1920]) # fix the y axis to the size of the screen display
    ax.set_ylim([0, 1080]) # fix the y axis to the size of the screen display
    ax.set_ylim(ax.get_ylim()[::-1])

    # highlight reported words
    dfWords = page.dfWords
    clicked_words = dfWords[dfWords['is_clicked']==1]

    for _, word in clicked_words.iterrows():
        # define the bounding box of the word
            word_x_start_img_pix = word['center_x'] - (word['width'] / 2)
            word_x_end_img_pix = word['center_x'] + (word['width'] / 2)
            word_y_start_img_pix = word['center_y'] - (word['height'] / 2)
            word_y_end_img_pix = word['center_y'] + (word['height'] / 2)

            # Convert image word pixels to eyelink pixels
            word_x_start,word_y_start = convert_pixel_to_eyelink(word_x_start_img_pix, word_y_start_img_pix)
            word_x_end, word_y_end = convert_pixel_to_eyelink(word_x_end_img_pix, word_y_end_img_pix)
           
            word_color = "#f77959" # red-ish color
            ax.add_patch(plt.Rectangle((word_x_start, word_y_start), 
                                        word_x_end-word_x_start, 
                                        word_y_end-word_y_start, 
                                        alpha=0.5, color=word_color))

    

def plot_fixations(page, ax, eye):
    '''
    Plots eye fixations as circles on the reading page. The color represents the
    occurrence timing, and the size of each circle is proportional to the fixation 
    duration. Fixations within a valid time window based on the window type (`win_type`)
    are plotted with different colors to represent whether they are from the 
    "Normal Reading" (NR) as blue or "Mindless Reading" (MR) as red condition.

    Parameters
    ----------
    page : object
        The page object containing fixation data. See `page.py` for more details on its attributes.
    ax : matplotlib.axes.Axes
        The axes on which to plot the fixations.
    eye : string
        Specifies which eye's fixation data to plot. 'L' for left eye and 'R' for right eye.

    Returns
    -------
    None
        The function modifies the given `ax` object in place by plotting the fixation points
        and their connecting lines.

    Notes
    -----
    This function uses two different sets of fixation data:
    - `dfAllFix`: Contains all fixation data for the page.
    - `dfFix`: Contains fixation data specific to a defined time window (e.g., NR vs. MR).
    The fixations from `dfFix` are colored red for MR (Reported) and blue for NR (Non-Reported),
    and their sizes are scaled down for better visibility.
    '''
    # get all fixation data from page object
    dfFix = page.dfAllFix
    dfFix = dfFix[dfFix['eye']==eye]
    # gaze pos x
    x_data = np.array(dfFix['xAvg'])
    # gaze pos y
    y_data = np.array(dfFix['yAvg'])
    # pupil size
    data_size = np.array(dfFix['duration'])/3
    
    # scatter plot the fixations as dots
    ax.scatter(x_data, y_data, c=np.arange(1,len(x_data)+1), s=data_size, alpha=0.7)
    
    # plot a line between two fixations
    for index in np.arange(1, len(x_data)):
        x = [x_data[index-1], x_data[index]]
        y = [y_data[index-1], y_data[index]]
        ax.plot(x, y, 'r', alpha=0.2)
    
    
    # get fixation data from defined time window
    dfFix = page.dfFix
    dfFix = dfFix[dfFix['eye']==eye]
    # gaze pos x
    x_data = np.array(dfFix['xAvg'])
    # gaze pos y
    y_data = np.array(dfFix['yAvg'])
    # pupil size
    data_size = np.array(dfFix['duration'])/3
    
    # define color: red for MR and blue for NR
    color = 'r' if page.mw_reported else 'b' 
    # differentiate fixations within the reported time window
    ax.scatter(x_data, y_data, c=color, s=data_size/5, alpha=0.9)

    
# def plot_individual_pages(page_index, pages):
#     '''
#     Handle function call for the individual plot
#     input:  pages - array, page objects
#             page_index - int, page to plot
#     output: NONE - show plot
#     '''

#     if page_index == 'all':
#         for i in range(len(pages)):
#             fig, ax1 = plt.subplots(1,1, figsize = (18, 18))
#             page_index = i
#             image_files = f'{READING_DIR}/{pages[page_index].image_file_location}'
#             ra.no_animate(image_files, pages[i], ax1)
#             plt.show()
#     else:
#         page_index = int(page_index)
#         fig, ax1 = plt.subplots(1,1, figsize = (18, 18))
#         image_files = f'{READING_DIR}/{pages[page_index].image_file_location}'
#         ra.no_animate(image_files, pages[page_index], ax1)
#         plt.show()
        

""" def plot_animation_page(page_index, pages):
    '''
    Handle function call for the animating plot
    input:  pages - array, page objects
            page_index - int, page to plot
    output: NONE - show plot
    '''
    fig, ax1 = plt.subplots(1,1, figsize = (18, 18))
    image_files = f'{READING_DIR}/{pages[page_index].image_file_location}'
    ani = animation.FuncAnimation(fig, ra.animate, interval=20, blit=False, fargs=(ax1, page_index, image_files, pages))
    plt.show() """


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
    for _, dfWords in page_data.dfWordss.iterrows():

        # define the bounding box of the word
        word_x_start_img_pix = dfWords['center_x'] - (dfWords['width'] / 2)
        word_x_end_img_pix = dfWords['center_x'] + (dfWords['width'] / 2)
        word_y_start_img_pix = dfWords['center_y'] - (dfWords['height'] / 2)
        word_y_end_img_pix = dfWords['center_y'] + (dfWords['height'] / 2)

        # Convert image word pixels to eyelink pixels
        word_x_start,word_y_start = convert_pixel_to_eyelink(word_x_start_img_pix, word_y_start_img_pix)
        word_x_end, word_y_end = convert_pixel_to_eyelink(word_x_end_img_pix, word_y_end_img_pix)

        # update color of highlight
        if (dfWords['is_error']): # if is_error
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
    for _, dfWords in page_data.dfWordss.iterrows():

        # define the bounding box of the word
        word_x_start_img_pix = dfWords['center_x'] - (dfWords['width'] / 2)
        word_x_end_img_pix = dfWords['center_x'] + (dfWords['width'] / 2)
        word_y_start_img_pix = dfWords['center_y'] - (dfWords['height'] / 2)
        word_y_end_img_pix = dfWords['center_y'] + (dfWords['height'] / 2)

        # Convert image word pixels to eyelink pixels
        word_x_start,word_y_start = convert_pixel_to_eyelink(word_x_start_img_pix, word_y_start_img_pix)
        word_x_end, word_y_end = convert_pixel_to_eyelink(word_x_end_img_pix, word_y_end_img_pix)

        # update color of highlight
        if (dfWords['is_error']): # if is_error
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
    