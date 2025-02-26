# class representation of a single page and all the details of that page

# Created 05/22 by GS
# Updated 10/10/23 by HS - turn dfWordss and fixations into pd dataframe
#                         - simplify fixation match
# Updated 11/1/23 by HS - fix the bug related df when matching fixation to word
# Updated 11/9/23 by HS - add 'zipf' to self.fixations for direct zipf-duration
#                         calculation 
#                       - new fileds to store information about reported MW
#                       - new function to extract self.feature_data only from
#                         reported mind-wandering time window
# updated 2/13/24 by HS - store blinks, fixatoins, and saccades info directly
#                         into dataframe instead of dict 
# updated 2/19/24 by HS - use samples for pupil size info
# updated 11/6/24 by HS - simplify page class and take out all feature calculations

import re
import string # to remove punctuation to find zipf scores.
import numpy as np
import pandas as pd
from textblob import TextBlob
from .match_word import find_match
from .utils import truncate_df_by_time, create_zipf_dict, convert_eyelink_to_image_pixel
#import reading_analysis as ra

class Page():

    def __init__ (self, start_time, imageLocation, page_view):
        
        # timestamp when the task starts
        self.task_start = np.nan
        # is MW reported. By default it is false
        self.mw_reported = False
        # reported mind-wandering onset and offset in seconds
        self.mw_onset = np.nan
        self.mw_offset = np.nan
        self.mw_dur = np.nan
        self.mw_valid = False
        
        # time window to extract eye features
        self.win_start = np.nan
        self.win_end = np.nan
        self.win_dur = np.nan

        # image file location saved locally
        self.image_file_location = imageLocation # page folder and title information
        self.reading = None
        self.page_view = page_view # '1st Pass' or 'Select Error' or '2nd Pass'
        self.run_number = np.nan
        self.page_number = np.nan # parsed from file lcoation infromation. page number that is being viewed on image
        self.time_start = start_time
        self.time_end = np.nan
        self.page_dur = np.nan

        # actual words displayed on the image page
        # dataframe adapted from reading coordinate file
        # Columns: 
        # words, width, height, center_x, center_y, is_clicked, ...
        # word_fix_times, fix_duration, sensetivity, zipf (in eyelink pixels)
        self.dfWords = None
        
        # dataframe adapted from saved fixation csv file
        # Columns: 
        # tStart, tEnd (miliseconds), duration, xAvg, yAvg, pupilAvg, fixed_word, fixed_word_index, zipf
        self.dfFix = None
        
        # dataframe adapted from saved blink csv file
        # Columns: 
        # tStart, tEnd (miliseconds), duration
        self.dfBlink = None
        
        # dataframe adapted from saved saccade csv file
        # Columns: 
        # tStart, tEnd (miliseconds), duration, xStart, yStart, 
        # xEnd, yEnd, ampDeg, vPeak
        self.dfSacc = None
        
        # dataframe adapted from saved sample csv file
        # Columns: 
        # tSample, LX, LY, LPupil, RX, RY, RPupil
        self.dfSamples = None
        self.parse_file_location(imageLocation)
        

    def parse_file_location(self, imageLocation):
        '''
        _summary_

        Args:
            imageLocation (_type_): _description_
        '''        
        # 02/22/23 Example imageLocation format: 
        # the_voynich_manuscript/the_voynich_manuscript_control/
        # the_voynich_manuscript_control_page01.png
        locationValues = imageLocation.split('/') 
        
        # extract reading article
        self.reading = locationValues[0]
        
        # use regular expression to extract page number
        self.page_number = int(re.findall(r"[-+]?(?:\d*\.\d+|\d+)", imageLocation)[-1])



    def calculate_duration(self):
        self.page_dur = self.time_end - self.time_start # all times in seconds
        
    
    def assign_data(self, dfFix, dfBlink, dfSacc, dfSamples):
        dfSamples, dfFix, dfSacc, dfBlink = truncate_df_by_time(dfSamples, dfFix, dfSacc, dfBlink, 
                                                                self.time_start, self.time_end)
        self.dfFix = dfFix
        self.dfBlink = dfBlink
        self.dfSacc = dfSacc
        self.dfSamples = dfSamples


    def load_word(self, task_path='../../MindlessReading/Reading', zipf_dict=None):
        # TODO
        '''
        _summary_

        Args:
            task_path (str, optional): _description_. Defaults to '../../MindlessReading/Reading'.
            zipf_dict (_type_, optional): _description_. Defaults to None.
        '''  
        # make the file name using the Page reference
        image_path = self.image_file_location.split('/')
        filename = f'{task_path}/{image_path[0]}/{image_path[1]}/{image_path[1]}_coordinates.csv'

        # read in the coordinate file as pd dataframe
        coordinate_df = pd.read_csv(filename)
        page_specific_df = coordinate_df[coordinate_df['page'] == self.page_number].copy()

        # add new columns fix_num and fix_dur to words pd and set default values to 0
        page_specific_df['fix_num'] = 0
        page_specific_df['fix_dur'] = 0.0

        # create zipf dict if not input
        if zipf_dict is None:
            zipf_dict = create_zipf_dict()
        
        # loop through each row to find the zipf score and save results in the column 'zipf'
        page_specific_df['zipf'] = np.nan
        for row_index, row in page_specific_df.iterrows():
            if isinstance(row['words'], str):
                word = row['words'].lower() # convert to lower-case
                word.translate(str.maketrans('', '', string.punctuation)) # remove punctuation
                # use try-except to ignore words cannot be recognized by textblob
                try:
                    word = TextBlob(word).words[0].singularize() # remove plurals
                except:
                    pass
                    
                if word in zipf_dict.keys():
                    page_specific_df.at[row_index, 'zipf'] = zipf_dict[word]

        # dfWordss is the dataframe
        self.dfWords = page_specific_df.reset_index()
        # add is_clicked column to the dataframe
        self.dfWords['is_clicked'] = 0
    

    def match_fix2words(self):
        '''
        match the fixation to the words on the current page

        intput: self - allows access to the variables
        output: None, does change the following object variables:
                self.dfFix - the fixations in the page
                self.dfWords - the actual word and the information about that word

        '''
        # declare three new columns and assign default values
        self.dfFix['fixed_word'] = 'NOT_FOUND'
        self.dfFix['fixed_word_index'] = float('inf')
        self.dfFix['zipf'] = np.nan
        self.dfFix['word_len'] = np.nan
        
        # loop through each fixation and match it to the closed word 
        for row_index, fix in self.dfFix.iterrows():
            # get the fixtaion position x and y
            x_eyelink = fix['xAvg']
            y_eyelink = fix['yAvg']
            # convert position x and y into image pixel unit
            fix_x, fix_y = convert_eyelink_to_image_pixel(x_eyelink, y_eyelink)
            
            # call function find the matched index
            matched_index = find_match(self.dfWords, (fix_x, fix_y))
            
            if matched_index >= 0:
                # store matched word info into fixations 
                word = self.dfWords['words'].iloc[matched_index]
                self.dfFix.at[row_index, 'fixed_word'] = word
                self.dfFix.at[row_index, 'fixed_word_index'] = matched_index
                self.dfFix.at[row_index, 'zipf'] = self.dfWords['zipf'].iloc[matched_index]
                word_len = len(word) if isinstance(word, str) else 0
                self.dfFix.at[row_index, 'word_len'] = word_len
                
                # store matched fixation info into dfWordss
                self.dfWords.at[matched_index, 'fix_num'] += 1
                self.dfWords.at[matched_index, 'fix_dur'] += fix['duration'] / 1000 # convert to seconds
        

                
    def find_MW_time(self, matched_words):
        '''
        This function finds time values for clicked words by looking at the first
        and the last fixation between two clicked words using Kadane's algorithm.
        
            Each fixation out of the MR window gets assigned a negative value (-1), 
            while fixation in the window gets assigned a positive value (+1.5). The
            function then finds the maximum subarray from this fixation value array.
            Time values for the first and the last fixation are then used as MR
            onset and offset. 

        Parameters
        ----------
        page : object reference
            DESCRIPTION. See page.py file for more details.
        matched_words : list of list (tuple)
            DESCRIPTION. len(matched_words) matches number of reported MW in csv file
            For each report, two words matched to two clicks are returned. 

        Returns
        -------
        None. But since page object is passed as a reference, fileds about the MW
        onset and offset will get modified after the function call 

        '''
    
        # get the page number
        page_num = self.page_number
        # set the mw_reported field to True
        self.mw_reported = True
        
        # get the index of word matched to click
        click1_word_index = matched_words['first_index'].iloc[page_num]
        click2_word_index = matched_words['second_index'].iloc[page_num]
        # make sure they are in order
        if click1_word_index > click2_word_index:
            click1_word_index, click2_word_index = click2_word_index, click1_word_index
        # update the clicked word (is_clicked = 1)
        self.dfWords.at[click1_word_index, 'is_clicked'] = 1
        self.dfWords.at[click2_word_index, 'is_clicked'] = 1
        
        # get indices of fixations between two reported words
        fix_mask = (self.dfFix['fixed_word_index'] >= click1_word_index) & \
                (self.dfFix['fixed_word_index'] <= click2_word_index)
        
        if np.sum(fix_mask) > 0:
            # convert it into int arrays:
            # -1: fixation out of the MR window
            # 1: fixation in the window
            fix_array = [1.5 if mw_fix else -1 for mw_fix in fix_mask]
            
            # declare variables to store information
            max_fix = float('-inf')
            current_fix = 0
            reset = True
            onset_ind = -1
            offset_ind = -1
            ind_dict = {'onset':-1, 'offset':-1}
            # use Kadane's algorithm to find the subarray containing maximum 
            # fixations in mind-wandering window
            for ind, fix_val in enumerate(fix_array):
                if reset:
                    onset_ind = ind
                current_fix += fix_val
                
                if current_fix >= max_fix:
                    max_fix = current_fix
                    offset_ind = ind
                    ind_dict['onset'] = onset_ind
                    ind_dict['offset'] = offset_ind
                
                if current_fix < 0:
                    current_fix = 0
                    reset = True
                else:
                    reset = False
            
            # onset
            # if the first word is selected, use the first fixations as onset
            if click1_word_index == 0:
                onset_ind = 0
            else:
                onset_ind = ind_dict['onset']
            tStart = self.dfFix['tStart'].iloc[onset_ind]
            self.mw_onset = tStart/1000
            
            # offset
            offset_ind = ind_dict['offset']
            tStart = self.dfFix['tStart'].iloc[offset_ind]
            self.mw_offset = tStart/1000
        
        self.mw_dur = self.mw_offset-self.mw_onset
        # check for valid mind-wandering onset and offset for the current page object
        if self.mw_dur >= 5:
            self.mw_valid = True
