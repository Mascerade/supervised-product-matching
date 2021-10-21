import pandas as pd
import os
import random
from itertools import combinations
from tqdm import tqdm
from src.common import create_final_data
from src.data_preprocessing import remove_misc, randomize_units
from supervised_product_matching.model_preprocessing import remove_stop_words

def generate_pos_pcpartpicker_data(df):
    '''
    Creates positive data from any of the PCPartPicker datasets
    '''

    columns = list(df.columns)
    pos_df = pd.DataFrame(columns=['title_one', 'title_two', 'label'])
    for idx in tqdm(range(len(df))):
        row = df.iloc()[idx]
        titles = []
        for col in columns:
            if not pd.isnull(row[col]): titles.append(remove_stop_words(row[col]))
        if len(titles) > 1:
            combs = combinations(titles, 2)
            for comb in combs:
                comb = list(comb)
                comb.append(1)
                pos_df = pos_df.append(pd.DataFrame([comb], columns=['title_one', 'title_two', 'label']))
    
    return pos_df

def generate_neg_pcpartpicker_data(df):
    '''
    Creates negative data from any of the PCPartPicker datasets
    '''

    columns = list(df.columns)
    neg_df = pd.DataFrame(columns=['title_one', 'title_two', 'label'])
    df_list = df.iloc()
    for idx in tqdm(range(len(df))):
        row = df_list[idx]
        for col in columns:
            if not pd.isnull(row[col]):
                neg_idx = None
                while neg_idx == idx or neg_idx is None:
                    neg_idx = random.randint(0, len(df) - 1)
                
                neg_title = None
                while neg_title == None or pd.isnull(neg_title):
                    neg_title = df_list[neg_idx][random.choice(columns)]
                
                neg_df = neg_df.append(pd.DataFrame([[remove_stop_words(row[col]), remove_stop_words(neg_title), 0]], columns=['title_one', 'title_two', 'label']))
    
    return neg_df

def create_pcpartpicker_data():
    '''
    Creates data for CPU, RAM, and drive data.
    Saves the data to final_pcpartpicker_data.csv
    '''
    
    file_path = 'data/train/final_pcpartpicker_data.csv'
    if not os.path.exists(file_path):
        print('Generating PCPartPicker data . . .')
        ram_df = remove_misc(pd.read_csv('data/base/pos_ram_titles.csv'))
        cpu_df = remove_misc(pd.read_csv('data/base/pos_cpu_titles.csv'))
        hard_drive_df = remove_misc(pd.read_csv('data/base/pos_hard_drive_titles.csv'))

        # Generate all the positive data for the categories
        pos_ram_data = generate_pos_pcpartpicker_data(ram_df)
        pos_cpu_data = generate_pos_pcpartpicker_data(cpu_df)
        pos_hard_drive_data = generate_pos_pcpartpicker_data(hard_drive_df)

        # Generate all the negative data for the categories
        neg_ram_data = generate_neg_pcpartpicker_data(ram_df)
        neg_cpu_data = generate_neg_pcpartpicker_data(cpu_df)
        neg_hard_drive_data = generate_neg_pcpartpicker_data(hard_drive_df)

        # Generate the final data
        final_ram_data = create_final_data(pos_ram_data, neg_ram_data)
        final_cpu_data = create_final_data(pos_cpu_data, neg_cpu_data)
        final_hard_drive_data = create_final_data(pos_hard_drive_data, neg_hard_drive_data)

        print('Amount of data for the CPU data, RAM data and drive data', len(final_cpu_data), len(final_ram_data), len(final_hard_drive_data))
        
        # Concatenate the data and save it
        final_pcpartpicker_df = pd.concat([final_ram_data, final_cpu_data, final_hard_drive_data])
        final_pcpartpicker_df.reset_index(inplace=True)
        randomize_units(final_pcpartpicker_df, units=['gb'])
        final_pcpartpicker_df.to_csv(file_path)

    else:
        print('Already have PCPartPicker data. Moving on . . .')
