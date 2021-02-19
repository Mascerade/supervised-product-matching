import pandas as pd
import os
from tqdm import tqdm
from src.common import Common, create_final_data
from src.preprocessing import remove_stop_words, randomize_units, replace_space_df

def preprocessing(orig_data):
    '''
    Normalizes the data by getting rid of stopwords and punctuation
    '''

    # Iterate over the original dataframe (I know it is slow and there are probably better ways to do it)
    iloc_data = orig_data.iloc
    
    # Will temporarily store the title data before it gets put into a DataFrame
    temp = []
    
    # Iterate over the data
    for idx in tqdm(range(len(orig_data))):
        row = iloc_data[idx]
        title_left = remove_stop_words(row.title_left)
        title_right = remove_stop_words(row.title_right)

        # Append the newly created row (title_left, title_right, label) to the the temporary list
        temp.append([title_left, title_right, row.label])
        
    # Return DataFrame of the title data, simplified
    return pd.DataFrame(temp, columns=Common.COLUMN_NAMES)

def create_train_df(df):
    '''
    Returns a shuffled dataframe with an equal amount of positive and negative examples
    '''

    # Get the positive and negative examples
    pos_df = df.loc[df['label'] == 1]
    neg_df = df.loc[df['label'] == 0]
    
    # Shuffle the data
    pos_df = pos_df.sample(frac=1)
    neg_df = neg_df.sample(frac=1)
    
    # Concatenate the positive and negative examples and 
    # make sure there are only as many negative examples as positive examples
    final_df = pd.concat([pos_df[:min(len(pos_df), len(neg_df))], neg_df[:min(len(pos_df), len(neg_df))]])
    
    # Shuffle the final data once again
    final_df.sample(frac=1)
    
    return final_df

def create_training_data(df, path):
    '''
    Creates and saves a simpler version of the original data that only contains the the two titles and the label.
    '''
    
    norm_bal_data = create_train_df(preprocessing(df))
    
    # Save the new normalized and simplified data to a CSV file to load later
    norm_bal_data.reset_index(inplace=True)
    randomize_units(norm_bal_data, units=['gb'])
    norm_bal_data.to_csv(path, index=False)


def create_computer_data():
    '''
    Simplifies the Gold Standard comptuer data and saves it to computers_train_bal_shuffle.csv
    '''
    
    computer_data_path = 'data/train/computers_train_bal_shuffle.csv'
    if not os.path.exists(computer_data_path):
        print('Generating simplifed Gold Standard computer data . . . ')
        # Load the data
        computer_df = pd.read_json('data/base/computers_train_xlarge_normalized.json.gz', compression='gzip', lines=True)    

        # Create and save the data if the simple and normalized data does not exist
        create_training_data(computer_df, computer_data_path)

    else: 
        print('Already have Gold Standard computer data. Moving on . . . ')
