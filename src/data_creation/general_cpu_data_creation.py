import pandas as pd
import os
from tqdm import tqdm
import random
from itertools import combinations
from supervised_product_matching.model_preprocessing import remove_stop_words
from src.common import create_final_data

def cpu_variations(cpu):
    '''
    Creates different forms of a cpu title
    Ex: amd ryzen 5 3600, amd ryzen 5 3600 6 core processor, 
    amd ryzen 5 3600 3.6 ghz processor and ryzen 5 3600 6 core 3.6 ghz processor.
    '''
    
    temp = []
    
    # This would be something like 'amd ryzen 5 3600 6 cores 3.6 ghz processor'
    temp.append(remove_stop_words('{} {} core {} processor'.format(cpu['name'], cpu['cores'], cpu['core_clock'])))

    # Just the base 'amd ryzen 5 3600'
    temp.append(remove_stop_words(cpu['name']))

    # Add in only cores 'amd ryzen 5 3600 6 core processor'
    temp.append(remove_stop_words('{} {} core processor'.format(cpu['name'], cpu['cores'])))

    # Add in only ghz 'amd ryzen 5 3600 3.6 ghz processor'
    temp.append(remove_stop_words('{} {} processor'.format(cpu['name'], cpu['core_clock'])))
    
    return temp

def generate_pos_cpu_data():
    '''
    Creates positive CPU data with different variations of the title that still 
    represent the same underlying CPU using the cpu_variations function.
    '''
    
    cpu_df = pd.read_csv('data/base/cpu_data.csv')
    cpu_df_iloc = cpu_df.iloc()
    pos_df = []
    
    # The data is (name, cores, core_clock)
    for idx in tqdm(range(len(cpu_df))):
        # For creating combinations
        temp = []
        cpu = cpu_df_iloc[idx]
        
        # Returns combos of the attributes of the CPU
        temp = cpu_variations(cpu)
        
        combs = list(combinations(temp, 2))
        for comb in combs:
            pos_df.append([comb[0], comb[1], 1])
        
    return pd.DataFrame(pos_df, columns=['title_one', 'title_two', 'label'])

def generate_neg_cpu_data():
    '''
    Creates negative CPU data that uses two different CPUs to create a pair. 
    '''

    cpu_df = pd.read_csv('data/base/cpu_data.csv')
    cpu_df_iloc = cpu_df.iloc()
    neg_df = []
    
    for idx in tqdm(range(len(cpu_df))):
        cpu = cpu_df_iloc[idx]
        key_brand = 'amd'
        
        # Placeholder for now
        neg_cpu = cpu
        
        if 'amd' in cpu['name'].lower():
            if random.random() > 0.65:
                key_brand = 'amd'
            else:
                key_brand = 'intel'

        elif 'intel' in cpu['name'].lower():
            if random.random() > 0.65:
                key_brand = 'intel'
            else:
                key_brand = 'amd'

        # Get something that is similar to it
        while key_brand not in neg_cpu['name'].lower() or cpu['name'] == neg_cpu['name']:
            neg_cpu = cpu_df_iloc[random.randrange(0, len(cpu_df))]
        
        orig_variations = cpu_variations(cpu)
        neg_variations = cpu_variations(neg_cpu)
                
        # Get all the combinations between orig variations and neg variations
        for orig_cpu in orig_variations:
            for neg_variation in neg_variations:
                neg_df.append([orig_cpu, neg_variation, 0])        
    
    return pd.DataFrame(neg_df, columns=['title_one', 'title_two', 'label'])

def create_general_cpu_data():
    '''
    Runs through generate_pos_cpu_data() and generate_neg_cpu_data() to create positive and negative data.
    Saves the file to more_cpu_data.csv
    '''
    
    file_path = 'data/train/more_cpu_data.csv'
    if not os.path.exists(file_path):
        print('Generating general cpu data . . . ')
        # Create the positive and negative examples
        pos_df = generate_pos_cpu_data()
        neg_df = generate_neg_cpu_data()

        # Concatenate the data and save it
        final_cpu_df = create_final_data(pos_df, neg_df)
        final_cpu_df.to_csv(file_path)

    else:
        print('Already have general cpu data data. Moving on . . .')
