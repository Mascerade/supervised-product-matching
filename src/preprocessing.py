import os
import sys
import pandas as pd
from nltk.corpus import stopwords
from tqdm import tqdm


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

# Drop the Unnamed: 0 column and drop any row where it is all NaN
def remove_misc(df):
    df = df.drop(columns=['Unnamed: 0'])
    df = df.dropna(how='all')
    return df
