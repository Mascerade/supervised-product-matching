import sys
import os
import pandas as pd

class Common():
    '''
    '''

    # Max length of a title to be fed into the model
    MAX_LEN = 44

    # The how many values are in each embedding vector
    EMBEDDING_SHAPE = (300,)

    # Number of training examples
    M = 19380

    # These are words that commonly come up with laptops
    MODIFIERS = ['premium', 'new', 'fast', 'latest model']
    ADD_INS = ['USB 3.0', 'USB 3.1 Type-C', 'USB Type-C', 'Bluetooth', 'WIFI', 'Webcam', 'FP Reader',
                'HDMI', '802.11ac', '802.11 ac', 'home', 'flagship', 'business', 'GbE LAN', 'DVD-RW',
                'DVD', 'Windows 10']

    # For creating laptop data
    HARD_DRIVE_TYPES = ['HDD', 'Hard Drive', 'Internal Hard Drive']
    SSD_TYPES = ['SSD', 'Solid State Drive', 'M.2 SSD', 'SATA SSD']

    # The column names for all the DataFrames
    COLUMN_NAMES = ['title_one', 'title_two', 'label']
    
def get_max_len(df):
    '''
    Gets the length of the largest string in a DataFrame.
    '''

    max_len = 0
    for row in df.itertuples():
        if len(row.title_one.split(' ')) > max_len:
            max_len = len(row.title_one.split(' '))
            
        if len(row.title_two.split(' ')) > max_len:
            max_len = len(row.title_two.split(' '))
    
    return max_len

def print_dataframe(df):
    '''
    Prints out the titles in a DataFrame.
    '''

    for idx in range(len(df)):
        print(df.iloc[idx].title_one + '\n' + df.iloc[idx].title_two)
        print('________________________________________________________________')

def create_final_data(pos_df, neg_df):
    '''
    Concatenates and shuffles positive and negative DataFrames and makes sure they are the same length.
    '''

    pos_df = pos_df.sample(frac=1)
    neg_df = neg_df.sample(frac=1)
    final_df = pd.concat([pos_df[:min(len(pos_df), len(neg_df))], neg_df[:min(len(pos_df), len(neg_df))]])
    final_df = final_df.sample(frac=1)
    return final_df
