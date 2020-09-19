import os
import sys
import pandas as pd
from nltk.corpus import stopwords
from tqdm import tqdm

# Have to download the stopwords
# nltk.download('stopwords')

# ## Data Processsing and Organization
# Here, all we really want to do is prepare the data for training. This includes:
# * Simplifying the original data
# * Normalizing the data 
# * Balancing the positive and negative examples
# * Creating the embedding representations that will actually get fed into the neural network
# Organizing and normalizing the data

def remove_stop_words(phrase):
    # Creates the stopwords
    to_stop = stopwords.words('english')
    punctuation = "!”#$%&’()*+,-./:;<=>?@[\]^_`{|}~ "
    for c in punctuation:
        to_stop.append(c)

    to_stop.append('null')
    
    for punc in punctuation:
        phrase = phrase.replace(punc, ' ')
    
    return ' '.join((' '.join([x for x in phrase.split(' ') if x not in to_stop])).split())

"""
Essentially, we want to only have three attributes for each training example: title_one, title_two, label
For normalization, we are just going to use the nltk stopwords and punctuation
"""

def preprocessing(orig_data):
    """
    Normalizes the data by getting rid of stopwords and punctuation
    """
    
    # The new names of the columns
    column_names = ['title_one', 'title_two', 'label']

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
    return pd.DataFrame(temp, columns=column_names)

def create_train_df(df):
    """
    Returns a shuffled dataframe with an equal amount of positive and negative examples
    """
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
    """
    Creates and saves a simpler version of the original data that only contains the the two titles and the label.
    """
    
    norm_bal_data = create_train_df(preprocessing(df))
    
    # Save the new normalized and simplified data to a CSV file to load later
    norm_bal_data.to_csv(path, index=False)
