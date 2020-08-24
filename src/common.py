import sys
import os
import fasttext
import pandas as pd

class Common():
    # Get the fasttext model (we are using the largest one they offer [600B tokens])
    print('Loading the fastText embeddings...')
    fasttext_model = fasttext.load_model('models/crawl-300d-2M-subword.bin')

    # Max length of a title to be fed into the model
    MAX_LEN = 43

    # The how many values are in each embedding vector
    EMBEDDING_SHAPE = (300,)

    # Number of training examples
    m = 19380

def create_final_data(pos_df, neg_df):
    pos_df.sample(frac=1)
    neg_df.sample(frac=1)
    final_df = pd.concat([pos_df[:min(len(pos_df), len(neg_df))], neg_df[:min(len(pos_df), len(neg_df))]])
    final_df = final_df.sample(frac=1)
    return final_df

def get_max_len(df):
    max_len = 0
    for row in df.itertuples():
        if len(row.title_one.split(' ')) > max_len:
            max_len = len(row.title_one.split(' '))
            
        if len(row.title_two.split(' ')) > max_len:
            max_len = len(row.title_two.split(' '))

    return max_len

def print_dataframe(df):
    for idx in range(len(df)):
        print(df.iloc[idx].title_one + '\n' + df.iloc[idx].title_two)
        print('________________________________________________________________')
