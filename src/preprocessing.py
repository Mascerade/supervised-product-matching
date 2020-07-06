import os
import sys
import pandas as pd
from nltk.corpus import stopwords

# Have to download the stopwords
# nltk.download('stopwords')

# ## Data Processsing and Organization
# Here, all we really want to do is prepare the data for training. This includes:
# * Simplifying the original data
# * Normalizing the data 
# * Balancing the positive and negative examples
# * Creating the embedding representations that will actually get fed into the neural network


# Organizing and normalizing the data
"""
Essentially, we want to only have three attributes for each training example: title_one, title_two, label
For normalization, we are just going to use the nltk stopwords and punctuation
"""

def preprocessing(orig_data):
    """
    Normalizes the data by getting rid of stopwords and punctuation
    """
    
    # Creates the stopwords
    to_stop = stopwords.words('english')
    punctuation = "!”#$%&’()*+,-./:;<=>?@[\]^_`{|}~ "
    for c in punctuation:
        to_stop.append(c)

    to_stop.append('null')
    
    # The new names of the columns
    column_names = ['title_one', 'title_two', 'label']
    # A new dataframe for the data we are going to be creating
    norm_computers = pd.DataFrame(columns = column_names)
    # Iterate over the original dataframe (I know it is slow and there are probably better ways to do it)
    for row in orig_data.itertuples():
        title_left = row.title_left.split(' ')
        title_right = row.title_right.split(' ')
        
        # Creates a new list of only elements that are not in the stop words
        temp_title_left = []
        for word in title_left:
            if word not in to_stop:
                temp_title_left.append(word)
                
        # Creates a new list of only elements that are not in the stop words
        temp_title_right = []
        for word in title_right:
            if word not in to_stop:
                temp_title_right.append(word)
        
        # Join the elements in the list to create the strings
        title_left = ' '.join(temp_title_left)
        title_right = ' '.join(temp_title_right)
        
        # Append the newly created row (title_left, title_right, label) to the new dataframe
        norm_computers = norm_computers.append(pd.DataFrame([[title_left, title_right, row.label]], columns=column_names))
        
    return norm_computers
        
def create_simple_data():
    """
    Creates and saves a simpler version of the original data that only contains the the two titles and the label.
    """

    # Get the dataset of computer parts
    computers_df = pd.read_json('data/computers_train/computers_train_xlarge_normalized.json.gz',compression='gzip', lines=True)
    norm_computers = preprocessing(computers_df)
    
    # Save the new normalized and simplified data to a CSV file to load later
    norm_computers.to_csv('data/computers_train/computers_train_xlarge_norm_simple.csv', index=False)

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
    final_df = pd.concat([pos_df, neg_df[:len(pos_df)]])
    
    # Shuffle the final data once again
    final_df.sample(frac=1)
    return final_df
