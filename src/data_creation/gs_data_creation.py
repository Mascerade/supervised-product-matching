import pandas as pd
import os
from tqdm import tqdm
from src.common import create_final_data, remove_stop_words
from src.common.Common import COLUMN_NAMES

"""
Essentially, we want to only have three attributes for each training example: title_one, title_two, label
For normalization, we are just going to use the nltk stopwords and punctuation
"""

def preprocessing(orig_data):
    """
    Normalizes the data by getting rid of stopwords and punctuation
    """
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
    return pd.DataFrame(temp, columns=COLUMN_NAMES)

def create_training_data(df, path):
    """
    Creates and saves a simpler version of the original data that only contains the the two titles and the label.
    """
    
    norm_bal_data = create_final_data(preprocessing(df))
    
    # Save the new normalized and simplified data to a CSV file to load later
    norm_bal_data.to_csv(path, index=False)


def create_computer_data():
    computer_data_path = 'data/train/computers_train_bal_shuffle.csv'
    if not os.path.exists(computer_data_path):
        # Load the data
        computer_df = pd.read_json('data/train/computers_train_xlarge_normalized.json.gz', compression='gzip', lines=True)    

        # Create and save the data if the simple and normalized data does not exist
        create_training_data(computer_df, computer_data_path)
