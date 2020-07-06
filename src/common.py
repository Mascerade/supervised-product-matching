import sys
import os
import fasttext

class Common():
    # Get the fasttext model (we are using the largest one they offer [600B tokens])
    fasttext_model = fasttext.load_model('models/crawl-300d-2M-subword.bin')

    # Max length of a title to be fed into the model
    MAX_LEN = 42

    # The how many values are in each embedding vector
    EMBEDDING_SHAPE = (300,)

    # Number of training examples
    m = 19380