import random
import re
from src.common import Common

# ## Data Processsing and Organization
# Here, all we really want to do is prepare the data for training. This includes:
# * Simplifying the original data
# * Normalizing the data 
# * Balancing the positive and negative examples
# * Creating the embedding representations that will actually get fed into the neural network
# Organizing and normalizing the data

def remove_misc(df):
    '''
    Drop the Unnamed: 0 column and drop any row where it is all NaN
    '''

    df = df.drop(columns=['Unnamed: 0'])
    df = df.dropna(how='all')
    return df

def replace_space(string, matches, unit, space=True):
    '''
    Randomly replace the the unit without a space or with a space
    '''
    
    for match in matches:
        match = match.strip()
        num = match.split(unit)[0].strip()
        if space:
            string = string.replace(match, '{} {}'.format(num, unit))
        else:
            string = string.replace(match, '{}{}'.format(num, unit))
    return string

def replace_space_df(df, units, space=True):
    # For each unit, do the replacement on it
    for unit in units:
        matcher = unit_matcher(unit)
        for idx in range(len(df)):
            title_one = df.at[idx, 'title_one']
            title_two = df.at[idx, 'title_two']
            title_one_matches = matcher.findall(title_one)
            title_two_matches = matcher.findall(title_two)
            df.at[idx, 'title_one'] = replace_space(title_one, title_one_matches, unit, space)
            df.at[idx, 'title_two'] = replace_space(title_two, title_two_matches, unit, space)

def unit_matcher(unit):
    return re.compile(' ?[0-9]+.{0,1}' + unit + '(?!\S)', re.IGNORECASE)

def randomize_units(df, units):
    """
    Replaces units like 8 gb with 8gb to have a better distribution across the dataset
    """
    
    # Randomly replace the the unit without a space or with a space 
    def random_replace(string, matches, unit):
        for match in matches:
            match = match.strip()
            num = match.split(unit)[0].strip()
            if random.random() < Common.NO_SPACE_RATIO:
                string = string.replace(match, '{}{}'.format(num, unit))
            else:
                string = string.replace(match, '{} {}'.format(num, unit))
        
        return string
    
    # For each unit, do the replacement on it
    for unit in units:
        matcher = unit_matcher(unit)
        for idx in range(len(df)):
            title_one = df.at[idx, 'title_one']
            title_two = df.at[idx, 'title_two']
            title_one_matches = matcher.findall(title_one)
            title_two_matches = matcher.findall(title_two)
            df.at[idx, 'title_one'] = random_replace(title_one, title_one_matches, unit)
            df.at[idx, 'title_two'] = random_replace(title_two, title_two_matches, unit)
