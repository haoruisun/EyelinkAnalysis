# ParseEyeLinkAsc.py
# - Reads in .asc data files from EyeLink and produces pandas dataframes for further analysis
#
# Created 7/31/18-8/15/18 by DJ.
# Updated 7/4/19 by DJ - detects and handles monocular sample data.
# Updated 10/30/24 HS - write a wrapper function that calls parse_EyeLinkAsc

# Import packages
import time
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from .page import Page

def parse_EyeLinkAsc(elFilename):
    # dfRec,dfMsg,dfFix,dfSacc,dfBlink,dfSamples = ParseEyeLinkAsc(elFilename)
    # -Reads in data files from EyeLink .asc file and produces readable dataframes for further analysis.
    #
    # INPUTS:
    # -elFilename is a string indicating an EyeLink data file from an AX-CPT task in the current path.
    #
    # OUTPUTS:
    # -dfRec contains information about recording periods (often trials)
    # -dfMsg contains information about messages (usually sent from stimulus software)
    # -dfFix contains information about fixations
    # -dfSacc contains information about saccades
    # -dfBlink contains information about blinks
    # -dfSamples contains information about individual samples
    #
    # Created 7/31/18-8/15/18 by DJ.
    # Updated 11/12/18 by DJ - switched from "trials" to "recording periods" for experiments with continuous recording

    # ===== READ IN FILES ===== #
    # Read in EyeLink file
    print('Reading in EyeLink file %s...'%elFilename)
    t = time.time()
    f = open(elFilename,'r')
    fileTxt0 = f.read().splitlines(True) # split into lines
    fileTxt0 = list(filter(None, fileTxt0)) #  remove emptys
    fileTxt0 = np.array(fileTxt0) # concert to np array for simpler indexing
    f.close()

    # Separate lines into samples and messages
    print('Sorting lines...')
    nLines = len(fileTxt0)
    lineType = np.array(['OTHER']*nLines,dtype='object')
    iStartRec = None
    t = time.time()
    for iLine in range(nLines):
        if len(fileTxt0[iLine])<3:
            lineType[iLine] = 'EMPTY'
        elif fileTxt0[iLine].startswith('*') or fileTxt0[iLine].startswith('>>>>>'):
            lineType[iLine] = 'COMMENT'
        elif fileTxt0[iLine].split()[0][0].isdigit() or fileTxt0[iLine].split()[0].startswith('-'):
            lineType[iLine] = 'SAMPLE'
        else:
            lineType[iLine] = fileTxt0[iLine].split()[0]
        if 'Reading START' in fileTxt0[iLine]:
            iStartRec = iLine+1


    # ===== PARSE EYELINK FILE ===== #
    t = time.time()
    # Trials
    print('Parsing recording markers...')
    iNotStart = np.nonzero(lineType!='START')[0]
    dfRecStart = pd.read_csv(elFilename,skiprows=iNotStart,header=None,sep=r'\s+',usecols=[1])
    dfRecStart.columns = ['tStart']
    iNotEnd = np.nonzero(lineType!='END')[0]
    dfRecEnd = pd.read_csv(elFilename,skiprows=iNotEnd,header=None,sep=r'\s+',usecols=[1,5,6])
    # TODO: Q: what are xRes and yRes values in elFile?
    dfRecEnd.columns = ['tEnd','xRes','yRes']
    # combine trial info
    dfRec = pd.concat([dfRecStart,dfRecEnd],axis=1)
    nRec = dfRec.shape[0]

    # Import Messages
    print('Parsing stimulus messages...')
    t = time.time()
    iMsg = np.nonzero(lineType=='MSG')[0]
    # set up
    tMsg = []
    txtMsg = []
    t = time.time()
    for i in range(len(iMsg)):
        # separate MSG prefix and timestamp from rest of message
        info = fileTxt0[iMsg[i]].split()
        # extract info
        tMsg.append(int(info[1]))
        txtMsg.append(' '.join(info[2:]))
    # Convert dict to dataframe
    dfMsg = pd.DataFrame({'time':tMsg, 'text':txtMsg})

    # Import Fixations
    print('Parsing fixations...')
    t = time.time()
    iNotEfix = np.nonzero(lineType!='EFIX')[0]
    dfFix = pd.read_csv(elFilename,skiprows=iNotEfix,header=None,sep=r'\s+',usecols=range(1,8))
    dfFix.columns = ['eye','tStart','tEnd','duration','xAvg','yAvg','pupilAvg']
    nFix = dfFix.shape[0]

    # Saccades
    print('Parsing saccades...')
    t = time.time()
    iNotEsacc = np.nonzero(lineType!='ESACC')[0]
    dfSacc = pd.read_csv(elFilename,skiprows=iNotEsacc,header=None,sep=r'\s+',usecols=range(1,11))
    dfSacc.columns = ['eye','tStart','tEnd','duration','xStart','yStart','xEnd','yEnd','ampDeg','vPeak']

    # Blinks
    print('Parsing blinks...')
    iNotEblink = np.nonzero(lineType!='EBLINK')[0]
    dfBlink = pd.read_csv(elFilename,skiprows=iNotEblink,header=None,sep=r'\s+',usecols=range(1,5))
    dfBlink.columns = ['eye','tStart','tEnd','duration']
    # print('Done! Took %f seconds.'%(time.time()-t))

    # determine sample columns based on eyes recorded in file
    eyesInFile = np.unique(dfFix.eye)
    if eyesInFile.size==2:
        print('binocular data detected.')
        cols = ['tSample', 'LX', 'LY', 'LPupil', 'RX', 'RY', 'RPupil']
    else:
        eye = eyesInFile[0]
        print('monocular data detected (%c eye).'%eye)
        cols = ['tSample', '%cX'%eye, '%cY'%eye, '%cPupil'%eye]
    # Import samples
    print('Parsing samples...')
    t = time.time()
    iNotSample = np.nonzero(np.logical_or(lineType!='SAMPLE', np.arange(nLines)<iStartRec))[0]
    dfSamples = pd.read_csv(elFilename,skiprows=iNotSample,header=None,sep=r'\s+',usecols=range(0,len(cols)))
    dfSamples.columns = cols
    # Convert values to numbers
    for eye in ['L','R']:
        if eye in eyesInFile:
            dfSamples['%cX'%eye] = pd.to_numeric(dfSamples['%cX'%eye],errors='coerce')
            dfSamples['%cY'%eye] = pd.to_numeric(dfSamples['%cY'%eye],errors='coerce')
            dfSamples['%cPupil'%eye] = pd.to_numeric(dfSamples['%cPupil'%eye],errors='coerce')
        else:
            dfSamples['%cX'%eye] = np.nan
            dfSamples['%cY'%eye] = np.nan
            dfSamples['%cPupil'%eye] = np.nan


    print('Done Parsing Data!')
    
    # Return new compilation dataframe
    return dfRec,dfMsg,dfFix,dfSacc,dfBlink,dfSamples


def load_data(file_path, is_overwrite=False):
    # TODO finish docstring
    '''
    _summary_

    Args:
        elFilename (_type_): _description_
        dataDir (_type_): _description_
        is_overwrite (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    '''    
    # Get the folder path
    dataDir = os.path.dirname(file_path)
    # Get the file name
    elFilename = os.path.basename(file_path)

    # Declare filenames
    elFileStart = os.path.splitext(elFilename)[0]
    # dataDir = "/home/george/Documents/GitHub/GlassBrain/reading-analysis/Examples" # folder where the data sits
    # elFilename = 'G_Run1_2022_03_03_11_08.asc' # filename of the EyeLink file (.asc)
    outDir = f'{dataDir}/{elFileStart}_data'
    
    # make the directory if not exists
    is_outDir_exists = os.path.exists(outDir)
    if not is_outDir_exists:
        os.mkdir(outDir)
        is_overwrite = True
    # check the folder contains all parsed files
    # reparse all eye features if missing any file
    else:
        if len(os.listdir(outDir)) != 6:
            is_overwrite = True
            
            
    # Make master list of dataframes to write (or read)
    allDataFrames = ['_']*6 # see following code about the dataframes
    allNames = ['Trial','Message','Fixation','Saccade','Blink','Sample'] # what they're called
    
    # parse eye features if overwrite is true
    if is_overwrite:
        print('Parse EyeLink .asc files and overwrite any existing parsed files...')
        # parse .asc file
        dfTrial,dfMsg,dfFix,dfSacc,dfBlink,dfSamples = parse_EyeLinkAsc(f'{dataDir}/{elFilename}')
        print('Saving results...')
    
        # Make master list of dataframes to write
        allDataFrames = [dfTrial,dfMsg,dfFix,dfSacc,dfBlink,dfSamples] # the dataframes
        allNames = ['Trial','Message','Fixation','Saccade','Blink','Sample'] # what they're called
    
        # Write dataframes to .csv files
        for i in range(len(allNames)):
            # save csv files
            outFilename = '%s/%s_%s.csv'%(outDir,elFileStart,allNames[i])
            print('   Saving %s output as %s...'%(allNames[i],outFilename))
            allDataFrames[i].to_csv(outFilename,float_format='%.1f',index=False)
    
    # else read in existing .csv files
    else:
        print('Directly read in parsed files...')
        for i in range(len(allNames)):
            # read in csv files
            outFilename = '%s/%s_%s.csv'%(outDir,elFileStart,allNames[i])
            allDataFrames[i] = pd.read_csv(outFilename)
            
        print('Done!')
    return allDataFrames
    #return dfTrial,dfMsg,dfFix,dfSacc,dfBlink,dfSamples


def generate_pages(dfMsg):
        '''
        intput: dfMsg - pandas dataframe of mesages from eyelink
        output: page_stamps - list of timestamps for each page start

        dfMesg has the following types of messages that we care about:
            - "TRIALID xx" where "xx" is the current page being displayed
            - "displayed error image xx" where "xx" is the image being displayed, this is the
                part of the experiment where the user is clicking on an error
            - "displayed image xx" where "xx" is the image being displayed, this is the part
                where the participant is actively doing the reading of the page
            - "display thought probe" which indicates the participant is doing a thought probe
            - "start error routine" which indicates the user is selecing the error and answering
                thought probe questions
        '''
        # declare list holder for page objects
        pages = []
        page_end_search = False # look for the end time of the page
        page_complete = False # look for the end time of the page
        page_view = '1st Pass'
        pages_time = dfMsg['time'].values
        pages_text = dfMsg['text'].values
        prev_message = 'IMNOTDEFINED'
        for index, message in tqdm(enumerate(pages_text), total=len(pages_text), desc="Processing Message DataFrame"):
            
            # looking for the timestamp when the task begins
            if 'Reading START' in message:
                task_start = pages_time[index]/1000

            # looking for image file. This is our new page start
            if 'TRIALID' in prev_message and '.png' in message: # this means an image is being viewed
                message_value = message.split(' ')
                page_reference = message_value[-1]
                # page_reference is imageLocation
                # eg. the_voynich_manuscript/the_voynich_manuscript_control/the_voynich_manuscript_control_page01.png
                new_page = Page(pages_time[index]/1000, page_reference, page_view) # convert time to seconds
                # look for the end time of the page
                page_end_search = True

            # now we're looking for the end time, so grab the next timestamp
            elif (('Current Page END' in message) and page_end_search):
                new_page.time_end = pages_time[index]/1000 # convert to seconds
                    
                new_page.calculate_duration() # calculate page duration in seconds
                # Now just start looking for first page again
                # this also flags system to append page to list
                page_end_search = False
                page_complete = True
                new_page.task_start = task_start

            # this means start and end times were found. append new page
            if (page_complete):
                new_page.load_word()           
                pages.append(new_page)
                page_complete = False

            # update previous message
            prev_message = message
            
        return pages
