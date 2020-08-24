import pandas as pd
import random
from itertools import combinations
from common import create_final_data

def remove_misc(df):
    # Drop the Unnamed: 0 column and drop any row where it is all NaN
    columns = list(df.columns)[1:]
    df = df.drop(columns=['Unnamed: 0'])
    df = df.dropna(how='all')
    print(len(df))
    return df

def generate_pos_pcpartpicker_data(df):
    columns = list(df.columns)
    pos_df = pd.DataFrame(columns=['title_one', 'title_two', 'label'])
    for idx in range(len(df)):
        row = df.iloc()[idx]
        titles = []
        for col in columns:
            if not pd.isnull(row[col]): titles.append(row[col])
        if len(titles) > 1:
            combs = combinations(titles, 2)
            for comb in combs:
                comb = list(comb)
                comb.append(1)
                pos_df = pos_df.append(pd.DataFrame([comb], columns=['title_one', 'title_two', 'label']))
    
    return pos_df

def generate_neg_pcpartpicker_data(df):
    columns = list(df.columns)
    neg_df = pd.DataFrame(columns=['title_one', 'title_two', 'label'])
    df_list = df.iloc()
    for idx in range(len(df)):
        row = df_list[idx]
        for col in columns:
            if not pd.isnull(row[col]):
                neg_idx = None
                while neg_idx == idx or neg_idx is None:
                    neg_idx = random.randint(0, len(df) - 1)
                
                neg_title = None
                while neg_title == None or pd.isnull(neg_title):
                    neg_title = df_list[neg_idx][random.choice(columns)]
                
                neg_df = neg_df.append(pd.DataFrame([[row[col], neg_title, 0]], columns=['title_one', 'title_two', 'label']))
    
    return neg_df

def create_pcpartpicker_data():
    ram_df = pd.read_csv('data/train/pos_ram_titles.csv')
    cpu_df = pd.read_csv('data/train/pos_cpu_titles.csv')
    hard_drive_df = pd.read_csv('data/train/pos_hard_drive_titles.csv')

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

    print('Amount of data for the CPU data, RAM data and hard drive data', len(final_cpu_data), len(final_ram_data), len(final_hard_drive_data))
    return final_cpu_data, final_ram_data, final_hard_drive_data