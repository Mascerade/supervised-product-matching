import pandas as pd
import os
import random
from src.data_creation.retailer_test_creation import create_pos_laptop_test_data
from src.data_preprocessing import unit_matcher, replace_space
from supervised_product_matching.model_preprocessing import remove_stop_words

ram = [2, 4, 8, 12, 16, 32, 64]
drive = [64, 128, 256, 512]

def replace_units(string, matches, unit, space=True):
    '''
    Change the unit to a different one
    '''
    
    match = random.choice(matches).strip()
    num = int(match.split(unit)[0].strip())
    orig = num
    if num < 100:
        while num == orig:
            num = random.choice(ram)
    else:
        while num == orig:
            num = random.choice(drive)

    if space:
        new_unit = '{} {}'.format(num, unit)
    else:
        new_unit = '{}{}'.format(num, unit)
    
    string = string.replace(match, new_unit)
    return string


def change_unit_retailer_data(df, units, space=True):
    """
    Replaces units like 8 gb with 8gb to have a better distribution across the dataset
    """
    temp = []
    
    # For each unit, do the replacement on it
    for unit in units:
        matcher = unit_matcher(unit)
        for idx in range(len(df)):
            for col in ['Amazon', 'Newegg', 'Walmart', 'BestBuy']:
                title = df.at[idx, col]
                if type(title) is str:
                    title = remove_stop_words(df.at[idx, col])
                    title_matches = matcher.findall(title)
                    if len(title_matches) > 0:
                        neg_title = replace_units(title, title_matches, unit, space)
                        title = replace_space(title, title_matches, unit, space)
                        
                        neg_matches = matcher.findall(neg_title)
                        neg_title = replace_space(neg_title, neg_matches, unit, space)
                        
                        temp.append([title, neg_title, 0])
                    
    return pd.DataFrame(temp, columns=['title_one', 'title_two', 'label'])

def change_unit_diff_titles(units, space=True):
    pos_laptop_df = pd.read_csv('data/base/retailer_test.csv')
    df = create_pos_laptop_test_data(pos_laptop_df)
    temp = []

    for unit in units:
        matcher = unit_matcher(unit)
        for idx in range(len(df)):
            title_one = df.iloc[idx].title_one
            title_two = df.iloc[idx].title_two

            title_one_matches = matcher.findall(title_one)
            title_two_matches = matcher.findall(title_two)
            if len(title_one_matches) > 0:
                new_title_one = replace_units(title_one, title_one_matches, unit, space)

            if (len(title_two_matches) > 0):
                new_title_two = replace_units(title_two, title_two_matches, unit, space)

            title_one = replace_space(title_one, title_one_matches, unit, space)
            title_two = replace_space(title_two, title_two_matches, unit, space)
            new_title_one_matches = matcher.findall(new_title_one)
            new_title_one = replace_space(new_title_one, new_title_one_matches, unit, space)
            new_title_two_matches = matcher.findall(new_title_two)
            new_title_two = replace_space(new_title_two, new_title_two_matches, unit, space)
            
            temp.append([title_one, new_title_two, 0])
            temp.append([new_title_one, title_two, 0])
    
    return pd.DataFrame(temp, columns=['title_one', 'title_two', 'label'])


def create_neg_laptop_test_data():
    '''
    Creates negative laptop test data simply by replacing hard drive attributes.
    Will use the unit 'gb' with and without a space after the number
    '''
    
    same_title_space_path = 'data/test/final_gb_space_laptop_test.csv'
    same_title_no_space_path = 'data/test/final_gb_no_space_laptop_test.csv'
    diff_title_space_path = 'data/test/final_retailer_gb_space_test.csv'
    diff_title_no_space_path = 'data/test/final_retailer_gb_no_space_test.csv'
    
    # Load the test laptop data
    laptop_df = pd.read_csv('data/base/retailer_test.csv')
    
    if not os.path.exists(same_title_space_path):
        print('Generating GB Laptop Test Data (With Space) . . . ')
        space_df = change_unit_retailer_data(laptop_df, ['gb'])
        space_df = space_df.sample(frac=1)
        space_df.to_csv(same_title_space_path)
        
    else:
        print('Already have GB Laptop Test Data (With Space). Moving on . . . ')
    
    if not os.path.exists(same_title_no_space_path):
        print('Generating GB Laptop Test Data (No Space) . . . ')
        no_space_df = change_unit_retailer_data(laptop_df, ['gb'], space=False)
        no_space_df = no_space_df.sample(frac=1)
        no_space_df.to_csv(same_title_no_space_path)
    
    else:
        print('Already have GB Laptop Test Data (No Space). Moving on . . . ')

    if not os.path.exists(diff_title_space_path):
        print('Generating Different Title GB Laptop Test Data (With Space) . . . ')
        df = change_unit_diff_titles(['gb'])
        df = df.sample(frac=1)
        df.to_csv(diff_title_space_path)

    else:
        print('Already have Different Title GB Laptop Test Data (With Space). Moving on . . .')

    if not os.path.exists(diff_title_no_space_path):
        print('Generating Different Title GB Laptop Test Data (No Space) . . . ')
        df = change_unit_diff_titles(['gb'], space=False)
        df = df.sample(frac=1)
        df.to_csv(diff_title_no_space_path)

    else:
        print('Already have Different Title GB Laptop Test Data (No Space). Moving on . . .')
