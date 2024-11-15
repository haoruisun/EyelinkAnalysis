# ParseEyeLinkAsc_script.ipynp
#
# Created 8/15/18 by DJ.

# Import packages
import os
import pandas as pd
import time
from ParseEyeLinkAsc import ParseEyeLinkAsc

# Declare filenames
dataDir = "../data/0000_00002022" # folder where the data sits
elFilename = 'DJ_2022_10_14_10_13.asc' # filename of the EyeLink file (.asc)
outDir = "."

# Navigate to data directory
os.chdir(dataDir)
# Load file in
dfTrial,dfMsg,dfFix,dfSacc,dfBlink,dfSamples = ParseEyeLinkAsc(elFilename)

print('Saving results...')
t = time.time()
# Get file prefix from original filename
elFileStart = os.path.splitext(elFilename)[0]
print(elFileStart)

# Make master list of dataframes to write
allDataFrames = [dfTrial,dfMsg,dfFix,dfSacc,dfBlink,dfSamples] # the dataframes
allNames = ['Trial','Message','Fixation','Saccade','Blink','Sample'] # what they're called
# Write dataframes to .csv files
for i in range(len(allNames)):
    outFilename = '%s/%s_%s.csv'%(outDir,elFileStart,allNames[i])
    print('   Saving %s output as %s...'%(allNames[i],outFilename))
    allDataFrames[i].to_csv(outFilename,float_format='%.1f',index=False)
print('Done! Took %f seconds.'%(time.time()-t))






