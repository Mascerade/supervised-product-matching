import pandas as pd
import numpy as np

""" LOCAL IMPORTS """
from src.data_preprocessing import remove_misc, randomize_units
from src.common import Common
from src.common import get_max_len, create_final_data
from src.data_creation.laptop_data_classes import populate_spec
from src.data_creation.general_cpu_data_creation import create_general_cpu_data
from src.data_creation.general_drive_data import create_final_drive_data
from src.data_creation.gs_data_creation import create_computer_gs_data
from src.data_creation.laptop_data_creation import create_pseudo_laptop_data
from src.data_creation.pcpartpicker_data_creation import create_pcpartpicker_data
from src.data_creation.retailer_test_creation import create_laptop_test_data
from src.data_creation.neg_laptop_test_creation import create_neg_laptop_test_data
from src.data_creation.retailer_laptop_train_creation import create_retailer_laptop_train_data

def gen_gb_pos_data():
    '''
    Create positive gigabyte data (ex: 8 gb vs 8 gb) to essentially 
    differentiate between numbers.
    '''
    pos = []
    for x in range(2, 5000, 2):
        attr = '{} {}'.format(x, 'gb')
        pos.append([attr, attr, 1])

    return pd.DataFrame(pos, columns = Common.COLUMN_NAMES)

def gen_neg_gb_data():
    '''
    Create negative gigabyte data (ex: 8 gb vs 10 gb) to essentially 
    differentiate between numbers.
    '''
    
    neg = []
    for x in range(2, 1000, 2):
        for y in range(2, 1000, 2):
            x_attr = '{} {}'.format(x, 'gb')
            y_attr = '{} {}'.format(y, 'gb')

            if x != y:
                neg.append([x_attr, y_attr, 0])

    return pd.DataFrame(neg, columns = Common.COLUMN_NAMES)

def create_data():
    '''
    Runs the necessary functions to create the data for training.
    '''
    
    # Don't show the copy warnings
    pd.set_option('mode.chained_assignment', None)

    # Run the functions
    populate_spec()
    create_pcpartpicker_data()
    create_general_cpu_data()
    create_final_drive_data()
    create_pseudo_laptop_data()
    final_gb_data = create_final_data(gen_gb_pos_data(), gen_neg_gb_data())
    final_gb_data.reset_index(inplace=True)
    randomize_units(final_gb_data, units=['gb'])
    create_laptop_test_data()
    create_neg_laptop_test_data()
    create_retailer_laptop_train_data()
    create_computer_gs_data()

    print('Generating gigabyte data (as in just examples that use GB)')

    # Load all the data
    final_computer_df = pd.read_csv('data/train/wdc_computers.csv')
    final_pseudo_laptop_df = pd.read_csv('data/train/spec_train_data_new.csv')
    final_pcpartpicker_data = pd.read_csv('data/train/final_pcpartpicker_data.csv').sample(frac=1)
    more_cpu_data = pd.read_csv('data/train/more_cpu_data.csv')
    more_drive_data = pd.read_csv('data/train/more_drive_data.csv')
    retailer_laptop_df = pd.read_csv('data/train/retailer_laptop_data.csv')
    all_data = [final_computer_df, final_pseudo_laptop_df, more_cpu_data, final_gb_data, more_drive_data, retailer_laptop_df]

    # Print the sizes of the data
    print('Computer df size: {}'.format(len(final_computer_df)))
    print('Pseudo-Laptop df size: {}'.format(len(final_pseudo_laptop_df)))
    print('PCPartPicker df size: {}'.format(len(final_pcpartpicker_data)))
    print('More Drive Data df size: {}'.format(len(more_drive_data)))
    print('More CPU Data df size: {}'.format(len(more_cpu_data)))
    print('Final GB Data: {}'.format(len(final_gb_data)))
    print('Retailer Laptop Data: {}'.format(len(retailer_laptop_df)))

    # Concatenate everything
    total_data = pd.concat(all_data)
    total_data = total_data.sample(frac=1)
    total_data = remove_misc(total_data)

    # Get the max length of the data for padding in BERT
    print('Max Length of Data: {}'.format(get_max_len(total_data)))

    print('Total data size: {}'.format(len(total_data)))

    # Save the data
    total_data.to_csv('data/train/total_data.csv', index=False)


if __name__ == "__main__":
    create_data()
