import math
import pandas as pd
import os
import random
from tqdm import tqdm
from src.preprocessing import remove_stop_words
from src.common import create_final_data
from itertools import combinations

def create_pos_laptop_test_data(laptop_df):
    '''
    Creates the positive test laptop data
    '''
    
    retailers = ['Amazon', 'Newegg', 'Walmart', 'BestBuy']
    pos_data = []
    for row in laptop_df.iloc:
        temp = []
        
        for retailer in retailers:
            if type(row[retailer]) is str:
                temp.append(row[retailer])
        
        temp = list(combinations(temp, 2))
        
        for combo in temp:
            combo = list(combo)
            combo[0] = remove_stop_words(combo[0]).lower()
            combo[1] = remove_stop_words(combo[1]).lower()
            pos_data.append(list(combo) + [1])
    
    return pd.DataFrame(pos_data, columns=['title_one', 'title_two', 'label'])


def create_neg_laptop_test_data(laptop_df):
    '''
    Creates the negative test laptop data
    '''

    retailers = ['Amazon', 'Newegg', 'Walmart', 'BestBuy']
    neg_data = []
    for row in laptop_df.iloc:
        temp = []
        
        for retailer in retailers:
            if type(row[retailer]) is str:
                orig_product = row[retailer]
                neg_product = ''
                
                while True:
                    idx = random.randint(0, len(laptop_df) - 1)
                    neg_row = laptop_df.iloc[idx]
            
                    rand_retailer = random.sample(retailers, 1)[0]
                    neg_product = neg_row[rand_retailer]
                    
                    if type(neg_product) is str and neg_product != orig_product:
                        temp = [remove_stop_words(orig_product).lower(), remove_stop_words(neg_product).lower(), 0]
                        neg_data.append(temp)
                        break
                        
                    else:
                        continue
    
    return pd.DataFrame(neg_data, columns=['title_one', 'title_two', 'label'])
    
def create_laptop_test_data():
    '''
    Creates positive and negative test laptop data and saves it to final_laptop_data.csv
    '''
    
    file_path = 'data/train/final_laptop_test_data.csv'

    # Load the test laptop data
    laptop_df = pd.read_csv('data/base/retailer_test.csv')
    
    if not os.path.exists(file_path):
        print('Generating test laptop data . . . ')

        # Create the negative and positive dataframes 
        neg_df = create_neg_laptop_test_data(laptop_df)
        pos_df = create_pos_laptop_test_data(laptop_df)
        
        # Concatenate the data and save it
        final_laptop_test_df = create_final_data(pos_df, neg_df)
        final_laptop_test_df = final_laptop_test_df.sample(frac=1)
        final_laptop_test_df.to_csv(file_path)

    else:
        print('Already have test laptop data. Moving on . . . ')