#!/usr/bin/env python

import fasttext
import pandas as pd
import numpy as np
import nltk
import os
import tensorflow as tf

from src.preprocessing import *
from src.embeddings import *
from src.model import *
from src.common import Common

# Create and save the data if the simple and normalized data does not exist
if not os.path.exists('data/computers_train/computers_train_xlarge_norm_simple.csv'):
    create_simple_data()

# Create and save the dataframe with equal numbers of positive and negative examples
# and is shuffled
if not os.path.exists('data/computers_train/computers_train_bal_shuffle.csv'):
    # Load the normal simplified data to create the balanced and shuffled data
    computer_df = pd.read_csv('data/computers_train/computers_train_xlarge_norm_simple.csv')
    create_train_df(computer_df).to_csv('data/computers_train/computers_train_bal_shuffle.csv', index=False)


# Load the saved balanced and shuffled data
df = pd.read_csv('data/computers_train/computers_train_bal_shuffle.csv')

# Save the embeddings from the data in the dataframe
save_embeddings(df, 'bal_embeddings', 'bal_labels')

print('Loading the embeddings and labels...')
# Load the embeddings and labels from the numpy files
embeddings, labels = load_embeddings_and_labels('bal_embeddings', 'bal_labels')

print('Splitting the data into train, validation and test...')
# Split the data into train, validation and test sets
X_train1 = embeddings[0, :15000]
X_train2 = embeddings[1, :15000]
X_train = np.stack((X_train1, X_train2))
print('Training shape: ' + str(X_train.shape))

X_val1 = embeddings[0, 15000:17000]
X_val2 = embeddings[1, 15000:17000]
X_val = np.stack((X_val1, X_val2))
print('Val shape: ' + str(X_val.shape))

X_test1 = embeddings[0, 17000:]
X_test2 = embeddings[1, 17000:]
X_test = np.stack((X_test1, X_test2))
print('Test shape: ' + str(X_test.shape))

Y_train = labels[:15000]
Y_val = labels[15000:17000]
Y_test = labels[17000:]

def convert_to_one_hot(Y, C):
    """
    Function to create the 
    """
    Y = np.eye(C)[Y.reshape(-1)]
    return Y

Y_train = convert_to_one_hot(Y_train.astype(np.int32), 2)
Y_val = convert_to_one_hot(Y_val.astype(np.int32), 2)
Y_test = convert_to_one_hot(Y_test.astype(np.int32), 2)

print('Creating the model graph...')
model = siamese_network((Common.MAX_LEN, Common.EMBEDDING_SHAPE[0],))
model.summary()

print('Compiling the model...')
# Compile the model
lr = 0.001
opt = tf.keras.optimizers.Adam(learning_rate=lr)
model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])

print('About to train...')
# Train the model
model.fit(x=[X_train1, X_train2], y=Y_train, batch_size=64, epochs=50, validation_data=([X_val[0], X_val[1]], Y_val))

# Test the model
results = model.evaluate([X_test1, X_test2], Y_test, batch_size=16)
print('test loss, test acc: ', results)

# Save the model
model_name = 'Softmax-LSTM-_epochs_loss'
save_model(model, model_name)
