import pandas as pd
import os
import random
from src.data_preprocessing import randomize_units
from supervised_product_matching.model_preprocessing import remove_stop_words
from src.common import create_final_data, Common

def generate_pos_hard_drive_data():
    '''
    Creates positive data with the same drive size, but different modifiers.
    Ex: 10 gb internal hard drive vs  10 gb hdd.
    '''
    
    pos_df = []
    drives = ['{} GB'.format(x) for x in range(1, 3193)] + ['{} TB'.format(x) for x in range(1, 101)]
    for drive in drives:
        # For hard drives
        pos_df.append([remove_stop_words('{} {}'.format(drive, random.choice(Common.HARD_DRIVE_TYPES))),
                       remove_stop_words('{} {}'.format(drive, random.choice(Common.HARD_DRIVE_TYPES))),
                       1])
        
        # For SSDs
        pos_df.append([remove_stop_words('{} {}'.format(drive, random.choice(Common.SSD_TYPES))),
                       remove_stop_words('{} {}'.format(drive, random.choice(Common.SSD_TYPES))),
                       1])
    
    return pd.DataFrame(pos_df, columns=Common.COLUMN_NAMES)

def generate_neg_hard_drive_data():
    '''
    Creates negative data with different drives sizes.
    Ex: 10 gb ssd vs 20 gb ssd.
    '''
    
    neg_df = []
    drives = ['{} GB'.format(x) for x in range(8, 1001, 8)] + ['{} TB'.format(x) for x in range(1, 20)]
    
    for drive in drives:
        new_drive = drive
        
        while new_drive == drive:
            new_drive = random.choice(drives)
        
        orig_variations = []
        new_variations = []
        
        # For hard drive
        for x in Common.HARD_DRIVE_TYPES:
            orig_variations.append('{} {}'.format(drive, x))
            new_variations.append('{} {}'.format(new_drive, x))
        
        # For ssd
        for x in Common.SSD_TYPES:
            orig_variations.append('{} {}'.format(drive, x))
            new_variations.append('{} {}'.format(new_drive, x))
        
        for old in orig_variations:
            for new in new_variations:
                neg_df.append([remove_stop_words(old), remove_stop_words(new), 0])
        
        
    return pd.DataFrame(neg_df, columns=Common.COLUMN_NAMES)

def create_final_drive_data():
    '''
    Creates positive and negative drive data and saves it to more_drive_data.csv
    '''
    
    file_path = 'data/train/more_drive_data.csv'
    if not os.path.exists(file_path):
        print('Generating general drive data . . . ')
        # Generate the data
        pos_df = generate_pos_hard_drive_data()
        neg_df = generate_neg_hard_drive_data()

        # Concatenate the data and save it
        final_df = create_final_data(pos_df, neg_df)
        final_df.reset_index(inplace=True)
        randomize_units(final_df, ['gb'])
        final_df.to_csv(file_path)

    else:
        print('Already have general drive data. Moving on . . .')
