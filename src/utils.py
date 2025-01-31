#
# Created 8/15/18 by DJ.
# Modified 10/26/22 by HS -- Update image dimensions
#                         -- Implement a dataframe in words_of_pages
# Modified 10/10/23 by HS - update words_of_pages function
# Updated on 10/31/23 by HS - directly read in eye features if they have alraedy
#                           - been parsed and saved
# Updated on 2/13/24 by HS - use eye samples for pupil analyses
# Updated 4/16/24 by HS - interpolate pupil size during blink
# 
# New script name: utils.py
# The script now contains all helper functions that parse and load data, analyze features, and 
# convert units among different coordinate systems. 
# Created 11/7/24 by HS

# Import packages
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d, CubicSpline

def truncate_df_by_time(page, win_start, win_end):
    # TODO
    '''
    _summary_

    Args:
        page (_type_): _description_
        win_start (_type_): _description_
        win_end (_type_): _description_
    '''    

    # blinks
    dfBlink = page.dfBlink
    blink_indices = (dfBlink['tStart'] >= win_start*1000) & \
                    (dfBlink['tEnd'] <= win_end*1000)
    page.dfBlink = dfBlink.loc[blink_indices].copy()

    # fixatoins
    dfFix = page.dfFix
    fix_indices = (dfFix['tStart'] >= win_start*1000) & \
                    (dfFix['tEnd'] <= win_end*1000)
    page.dfFix = dfFix.loc[fix_indices].copy()

    # saccades
    dfSacc = page.dfSacc
    sacc_indices = (dfSacc['tStart'] >= win_start*1000) & \
                    (dfSacc['tEnd'] <= win_end*1000)
    page.dfSacc = dfSacc.loc[sacc_indices].copy()
    
    # eye samples (pupil info)
    dfSamples = page.dfSamples
    pupil_indices = (dfSamples['tSample'] >= win_start*1000) & \
                    (dfSamples['tSample'] <= win_end*1000)
    page.dfSamples = dfSamples.loc[pupil_indices].copy()



def interpolate_blink(dfSamples, dfBlink, dfSaccade):
    """
    Interpolate left and right pupil sizes over blink periods. Modifies the
    dataframe of samples in place to change pupil dilation values to interpolated
    values, effectively removing blink artifacts. Saves interpolated data as csv.
    
    Uses saccades as t1 and t4. Contains adjustments recommended through conversation
    with Dr. J. Performs the interpolation over the normalized pupil dilation values.
    
    Inputs:
        - dfSamples: A dataframe containing samples for all subjects and all runs
        - dfBlink: A dataframe containing information about the eye in which a 
            blink occured and the time that that blink occured
        - dfSaccades: A dataframe of saccade events
        
    Returns:
        Interpolated dfSamples
    """
    # extracted from reading_analysis.py (author: HS)
    # interpolate the pupil size during the blink duration
    # http://dx.doi.org/10.6084/m9.figshare.688002
    

    # get time array from dfSamples
    sample_time = dfSamples['tSample'].to_numpy()

    # interpolate data for LEFT and RIGHT eye separately
    for eye in ['L', 'R']:
        # extract blink and saccade information for one eye
        dfBlink_ = dfBlink[dfBlink['eye']==eye]
        dfSaccade_ = dfSaccade[dfSaccade['eye']==eye]

        # truncate blink dataframe using the saccade information
        t_start = dfSaccade_['tStart'].min()
        t_end = dfSaccade_['tEnd'].max()
        mask = (dfBlink_['tStart'] > t_start) & (dfBlink_['tEnd'] < t_end)
        dfBlink_ = dfBlink_[mask]

        # convert df columns to np.arrays for interpolation
        col_names = [f'{eye}X', f'{eye}Y', f'{eye}Pupil']
        data_to_interpolate = []
        for col_name in col_names:
            data_to_interpolate.append(np.array(dfSamples[col_name]))

        # iterate throu each row of blink dataframe
        for index in np.arange(len(dfBlink_)):
            row = dfBlink_.iloc[index]
            # get the start and end time
            b_start = row['tStart'] 
            b_end = row['tEnd']
            # skip blinks out of range of dfSamples
            if (b_start < sample_time[0]) and (b_end > sample_time[-1]):
                continue

            # set t1 to be the end time of the last saccade before the blink
            #get all saccades before this blink
            previous_sac = dfSaccade_[dfSaccade_["tEnd"] < b_start]
            # get last saccade before this blink
            t1 = previous_sac["tEnd"].max()
            # set t4 to be the start time of the first saccade after the blink
            # get all saccades after this blink
            after_sac = dfSaccade_[dfSaccade_["tStart"] > b_end]
            # get the first saccade after this blink
            t2 = after_sac["tStart"].min()

            # check for missing vals in t1 or t4 and use fallback if needed
            # if pd.isna(t1) or pd.isna(t2):
            #     raise ValueError("t1/t2 are Na")
            
            # check the timing of saccades are within the time array for samples
            if (t1 > sample_time[0]) and (t2 < sample_time[-1]):
                # choose data points for interpolation function
                x = [t1,t2]
                y_ind = []
                for t in x:
                    y_ind.append(np.where(sample_time==t)[0][0])

                # loop thru all columns
                for col_name, col_data in zip(col_names, data_to_interpolate):
                    # create the 1D function for interpolation
                    y = col_data[y_ind]
                    interp_f = interp1d(x, y)           
                    #spl = CubicSpline(x, y)
                    
                    # generate mask for blink duration
                    mask = (sample_time > t1) & (sample_time < t2)
                    time_to_interpolate = sample_time[mask]
                    # use spl model to interpolate data during blink duration
                    interp_data = interp_f(time_to_interpolate)
                    
                    # update the dfSamples in place
                    dfSamples.loc[mask, col_name] = interp_data

    return dfSamples


def downsample_data(df, downsample_factor=10):
    '''
    downsample size input dataframe to facilitate feature calcuations

    Args:
        df (dataframe or dataframe list): dataframe to be downsampled
        downsample_factor (int, optional): downsampling factor. Defaults to 10.

    Returns:
        dataframe or dataframe list: downsampled dataframe
    '''
    # check if input var is a list of dataframe
    if isinstance(df, list):
        for each_df in df:
            each_df = each_df.iloc[::downsample_factor, :]
    else:
        df = df.iloc[::downsample_factor, :]
    
    # return downsampled dataframe/list
    return df

            

def create_zipf_dict(zipf_filename = './res/word_sensitivity_table.xlsx'):
    # TODO
    '''
    _summary_

    Args:
        zipf_filename (str, optional): _description_. Defaults to '../res/word_sensitivity_table.xlsx'.

    Returns:
        _type_: _description_
    '''    
    return pd.read_excel(zipf_filename, usecols=['Word', 'FreqZipfUS']).set_index('Word').to_dict()['FreqZipfUS']

    
# The following functions convert coordinate among different systems (PsychoPy Height Unit, Image Pixel, Relative Position). 
# Refer to this post for understanding: https://wordpress.com/post/glassbrainlab.wordpress.com/623

def convert_error_page_pixel_to_py(x_image, y_image):
    '''
    Convert error page pixel to psychopy height unit. Note that error page is not the same as normal reading page nor the whole screen.

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


def convert_pixel_to_eyelink(x_image,y_image):
    """
    converts image pixels to eyelink pixels

    inputs:

    x_image: float
    The term to convert x coordinates to eye link

    y_image: float
    The term to convert y coordinates to eyelink
​
    output:

    x_eyelink : float
    the converted x coordinate

    y_eyelink: float
    the converted y coordinate

    (Assuming we are converting from the dimensions 1209x918 to 1418x1070)(x,y)
    The dimensions for updated images: 1900x1440
    """

    #changes x to eyelink coordinates
    # x_eyelink = (1418/1209)*x_image+355.2
    x_eyelink = (1080 * 1.3/1900)*x_image+258

    #changes y to eyelink coordinates
    # y_eyelink = (1070/918)*y_image+27
    y_eyelink = (1080 * 0.99/1442)*y_image+5.4

    return x_eyelink,y_eyelink


def convert_py_to_eyelink( x_psycho , y_psycho ):
    """
    Converts py to eyelink terms

    Inputs:

    psycho_x float
    The term to convert psychopy x-coordinates

    psycho_y: float
    The term the psychopy y-coordinates

    output:

    x_pixel: float
    The converted x coordinate [py to eyelink]

    y_pixel: float
    the converted y coordinate [py to eyelink]

        (Assuming we are converting from the dimensions 1.12x0.85 to 1418x1070 (x,y)
    11/07/22 Updated: 1.3 x 0.99 to 1900x1442
     """

    #changes x to from psychopy to eyelink coordinates
    x_eyelink = (1209/1.3)*x_psycho+960

    #changes y to from psychopy to eyelink coordinates
    y_eyelink = -(918/0.99)*y_psycho+540

    return  x_eyelink, y_eyelink


def convert_pixel_to_py(x_image, y_image):
    '''
    Convert image pixel to psychopy height unit

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
    x_py = (x_image-1900/2)/1900 * 1.3
    y_py = (y_image-1440/2)/1440 * 0.99
    
    return x_py, y_py


def convert_eyelink_to_image_pixel(x_eyelink, y_eyelink):
        '''
        Convert eyeylink coords to pixel for image (1900 x 1442 pixels) 
        displayed at pos (0, 0) w/ size (1.3, 0.99) in PsychoPy.

        Parameters
        ----------
        x_eyelink : float
            DESCRIPTION. eyelink coord unit
        y_eyelink : float
            DESCRIPTION. eyelink coord unit

        Returns
        -------
        x_pixel : float
            DESCRIPTION. Image pixel unit
        y_pixel : float 
            DESCRIPTION. Image pixel unit 

        '''
        x_pixel = (x_eyelink-258) * 1900 / (1080*1.3)
        y_pixel = (y_eyelink-5.4) * 1442 / (1080*0.99)
        return x_pixel, y_pixel
