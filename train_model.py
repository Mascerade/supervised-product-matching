#!/usr/bin/env python

import fasttext
import pandas as pd
import numpy as np
import nltk
import os
import tensorflow as tf

from src.preprocessing import remove_stop_words, preprocessing, create_train_df, create_training_data
from src.embeddings import load_embeddings_and_labels, save_embeddings, create_embeddings
from src.model import siamese_network, save_model
from src.common import Common, get_max_len
from src.laptop_data_creation import create_laptop_data
from src.pcpartpicker_data_creation import create_pcpartpicker_data

# If the computer data has not been made yet, make it
computer_data_path = 'data/train/computers_train_bal_shuffle.csv'
if not os.path.exists(computer_data_path):
    computer_df = pd.read_json('data/train/computers_train_xlarge_normalized.json.gz', compression='gzip', lines=True)
    create_training_data(computer_df, computer_data_path)

# Create and save the dataframe with equal numbers of positive and negative examples
# and is shuffled
if not os.path.exists('data/computers_train/computers_train_bal_shuffle.csv'):
    # Load the normal simplified data to create the balanced and shuffled data
    computer_df = pd.read_csv('data/computers_train/computers_train_xlarge_norm_simple.csv')
    create_train_df(computer_df).to_csv('data/computers_train/computers_train_bal_shuffle.csv', index=False)

if not os.path.exists('data/numpy_data/all_embeddings.npy'):
    # Load the computer data
    final_computer_df = pd.read_csv('data/train/computers_train_bal_shuffle.csv')

    # Create and get the laptop data
    final_laptop_df = create_laptop_data()

    # Create and get the PCPartPicker data
    final_cpu_df, final_ram_df, final_hard_drive_df = create_pcpartpicker_data()

    # Concatenate everything
    total_data = pd.concat([final_computer_df, final_laptop_df, final_cpu_df, final_ram_df, final_hard_drive_df])
    total_data = total_data.sample(frac=1)
    Common.MAX_LEN = get_max_len(total_data)
    save_embeddings(total_data, 'all_embeddings', 'all_labels')

print('Loading the embeddings and labels...')
# Load the embeddings and labels from the numpy files
embeddings, labels = load_embeddings_and_labels('all_embeddings', 'all_labels')

print('Splitting the data into train, validation and test...')

# Split the data into train, validation and test sets
X_train1 = embeddings[0, :len(labels) - 4000]
X_train2 = embeddings[1, :len(labels) - 4000]
X_train = np.stack((X_train1, X_train2))
print('Training shape: ' + str(X_train.shape))

X_val1 = embeddings[0, len(labels) - 4000:len(labels) - 2000]
X_val2 = embeddings[1, len(labels) - 4000:len(labels) - 2000]
X_val = np.stack((X_val1, X_val2))
print('Val shape: ' + str(X_val.shape))

X_test1 = embeddings[0, len(labels) - 2000:]
X_test2 = embeddings[1, len(labels) - 2000:]
X_test = np.stack((X_test1, X_test2))
print('Test shape: ' + str(X_test.shape))

Y_train = labels[:len(labels) - 4000]
print('Training labels shape:', str(Y_train.shape))

Y_val = labels[len(labels) - 4000:len(labels) - 2000]
print('Val shape:', str(Y_val.shape))

Y_test = labels[len(labels) - 2000:]
print('Test shape:', str(Y_test.shape))

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
