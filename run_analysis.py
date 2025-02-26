# -*- coding: utf-8 -*-
"""
This is the main function that calls moduels and extracts eye features.   

Created on 10/29/24 by HS

"""
import glob
import argparse
import src.extract_eye_features as ef

def main(sub_id='all', win_type='default'):
    # data_path = r"E:\MindlessReading\Data"
    data_path = r"Z:\Mindless Reading\Data"
    # Get list of subject folders or create single subject folder path
    if sub_id == 'all':
        # get all subject folders in the root path
        subject_folders = glob.glob(f'{data_path}/s[0-9]*')
    else:
        subject_folders = [f'{data_path}/s{sub_id}']
        
    # loop through individual subject
    for sub_folder in subject_folders:
        ef.extract_subject_features(sub_folder, win_type)

    return


if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Extract eye features for specified subjects.")
    parser.add_argument(
        "--sub_id", 
        type=str, 
        default="all", 
        help="Subject ID to extract features for (use 'all' for all subjects)"
    )
    parser.add_argument(
        "--win_type", 
        type=str, 
        default="default", 
        help="Window type for feature extraction"
    )

    # Parse arguments
    args = parser.parse_args()

    # Call main with parsed arguments
    main(sub_id=args.sub_id, win_type=args.win_type)