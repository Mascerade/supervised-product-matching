import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.common import Common

"""
Create the numpy files of all the training embedddings
We will have two numpy files:
1. The training/validation/test sets
2. The labels
"""

def create_embeddings(df):
    # Create the numpy arrays for storing the embeddings and labels
    total_embeddings = np.zeros(shape=(len(df), 2, Common.MAX_LEN, Common.EMBEDDING_SHAPE[0]))
    labels = np.zeros(shape=(len(df)))
    
    # I know this is a terrible way of doing this, but iterate over the dataframe
    # and generate the embeddings to add to the numpy array
    for idx, row in enumerate(tqdm(df.itertuples())):
        for word_idx, word in enumerate(row.title_one.split()):
            total_embeddings[idx, 0, word_idx] = Common.fasttext_model[word]
            
        for word_idx, word in enumerate(row.title_two.split()):
            total_embeddings[idx, 1, word_idx] = Common.fasttext_model[word]
            
        labels[idx] = row.label
        
    return total_embeddings, labels

def save_embeddings(df, embeddings_name, labels_name):
    """
    Saves the embeddings given the embeddings file name and labels file name
    """
    if not os.path.exists('data/numpy_data/' + embeddings_name + '.npy'):
        embeddings, labels = create_embeddings(df)
        with open('data/numpy_data/' + embeddings_name + '.npy', 'wb') as f:
            np.save(f, embeddings)

        with open('data/numpy_data/' + labels_name + '.npy', 'wb') as f:
            np.save(f, labels)

def load_embeddings_and_labels(embeddings_name, labels_name):
    loaded_embeddings = None
    labels = None
    with open('data/numpy_data/' + embeddings_name + '.npy', 'rb') as f:
        loaded_embeddings = np.load(f)
        loaded_embeddings = np.transpose(loaded_embeddings, (1, 0, 2, 3))
    
    with open('data/numpy_data/' + labels_name + '.npy', 'rb') as f:
        labels = np.load(f)
    
    return loaded_embeddings, labels
