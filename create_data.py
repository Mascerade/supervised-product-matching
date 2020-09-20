import pandas as pd
import numpy as np
import os

""" LOCAL IMPORTS """
from src.preprocessing import remove_misc
from src.common import Common
from src.common import get_max_len, COLUMN_NAMES, create_final_data
from src.data_creation.general_cpu_data_creation import create_general_cpu_data
from src.data_creation.general_drive_data import create_final_drive_data
from src.data_creation.gs_data_creation import create_computer_data
from src.data_creation.laptop_data_creation import create_laptop_data
from src.data_creation.pcpartpicker_data_creation import create_pcpartpicker_data
from src.data_creation.spec_creation import create_spec_laptop_data

# Run the functions
create_computer_data()
create_pcpartpicker_data()
create_general_cpu_data()
create_final_drive_data()
create_laptop_data()
create_spec_laptop_data()

print('Generating gigabyte data (as in just examples that use GB')

def gen_gb_pos_data():
    pos = []
    for x in range(2, 5000, 2):
        attr = '{} {}'.format(x, 'gb')
        pos.append([attr, attr, 1])

    return pd.DataFrame(pos, columns = COLUMN_NAMES)

def gen_neg_gb_data():
    neg = []
    for x in range(2, 1000, 2):
        for y in range(2, 1000, 2):
            x_attr = '{} {}'.format(x, 'gb')
            y_attr = '{} {}'.format(y, 'gb')

            if x != y:
                neg.append([x_attr, y_attr, 0])

    return pd.DataFrame(neg, columns = COLUMN_NAMES)

final_gb_df = create_final_data(gen_gb_pos_data(), gen_neg_gb_data())

# Load all the data
final_computer_df = pd.read_csv('data/train/computers_train_bal_shuffle.csv')
final_laptop_df = pd.read_csv('data/train/final_laptop_data.csv')
final_spec_df = pd.read_csv('data/train/spec_train_data.csv')[:15000]
final_pcpartpicker_data = pd.read_csv('data/train/final_pcpartpicker_data.csv').sample(frac=1)
more_cpu_data = pd.read_csv('data/train/more_cpu_data.csv')
more_drive_data = pd.read_csv('data/train/more_drive_data.csv')
all_data = [final_computer_df, final_laptop_df, final_spec_df, final_pcpartpicker_data, more_cpu_data, more_drive_data, final_gb_df]

print('Computer df size: {}'.format(len(final_computer_df)))
print('Laptop df size: {}'.format(len(final_laptop_df)))
print('Final spec df size: {}'.format(len(final_spec_df)))
print('PCPartPicker df size: {}'.format(len(final_pcpartpicker_data)))
print('More CPU df size: {}'.format(len(more_cpu_data)))
print('More drive df size: {}'.format(len(more_drive_data)))
print('GB df size: {}'.format(len(final_gb_df)))

# Concatenate everything
total_data = pd.concat(all_data)
total_data = total_data.sample(frac=1)
total_data = remove_misc(total_data)
Common.MAX_LEN = get_max_len(total_data)

print('Total data size: {}'.format(len(total_data)))

# Save the data
total_data.to_csv('data/train/total_data.csv')