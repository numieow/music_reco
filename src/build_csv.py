# Build a csv file by going through files in a folder
# and extracting the data from the files

import os
import csv
import re
import pandas as pd
import numpy as np

TEST_PATHS = ['Test/Part1/', 'Test/Part2/', 'Test/Part3/']
TRAIN_SUBFILES = ['cel', 'cla', 'flu', 'gac', 'gel', 'org', 'pia', 'sax', 'tru', 'vio', 'voi']
TRAIN_PATHS = ['Train' + '/' + subfile + '/' for subfile in TRAIN_SUBFILES]

def build_test_csv(folder_paths, csv_file_name):

    # Get the list of files in the folders
    files = [os.listdir(folder_path) for folder_path in folder_paths]

    # Create a csv file
    with open(csv_file_name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Write the header
        writer.writerow(['Track Name', 'Instrument'])

        # Go through the files and extract the data
        for i in range(len(folder_paths)):
            for file in files[i]:   
                if file.endswith('.txt'):
                    with open(folder_paths[i] + file, 'r') as f:
                        lines = f.readlines()
                        for line in lines:
                            file_name = folder_paths[i] + file.split('.txt')[0]
                            writer.writerow([file_name, line.strip()])

    # Read the csv file and return the dataframe
    df = pd.read_csv(csv_file_name)
    return df

def build_train_csv(folder_paths, csv_file_name):
    # Get the list of files in the folders
    files = [os.listdir(folder_path) for folder_path in folder_paths]

    # Create a csv file
    with open(csv_file_name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Write the header
        writer.writerow(['Track Name', 'Instrument'])

        # Go through the files and extract the data
        for i in range(len(folder_paths)):
            for file in files[i]:
                file = file.split('.wav')[0]
                first_bracket = file.find('[')
                last_bracket = file.rfind(']')

                file_name = file[0:first_bracket] + file[last_bracket+1:]
                
                instruments_init = file[first_bracket:last_bracket+1]
                instruments = instruments_init
                instrument_names = []

                while len(instruments) > 0:
                    open_brack = instruments.find('[')
                    close_brack = instruments.find(']')
                    name = instruments[open_brack+1:close_brack]
                    if name in TRAIN_SUBFILES:
                        instrument_names.append(name)
                    instruments = instruments[close_brack+1:]

                for instrument in instrument_names:
                    writer.writerow([file_name, instrument])

    # Read the csv file and return the dataframe
    df = pd.read_csv(csv_file_name)
    return df

#build_test_csv(TEST_PATHS, 'test_data.csv')
#build_train_csv(TRAIN_PATHS, 'train_data.csv')