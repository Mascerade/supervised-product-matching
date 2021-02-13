import pandas as pd
import numpy as np
import nltk
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModel, AdamW
import time

""" LOCAL IMPORTS """
from src.preprocessing import remove_misc, character_bert_preprocess_batch, bert_preprocess_batch
from src.common import Common, get_max_len
from create_data import create_data

using_model = "scaled characterbert add"

# Get the folder name in models
FOLDER = sys.argv[1]

# Get the model name from the terminal
MODEL_NAME = sys.argv[2]

print('\nOutputing models to {} with base name {}\n'.format(FOLDER, MODEL_NAME))

# Create the folder for the model if it doesn't already exist
if not os.path.exists('models/{}'.format(FOLDER)):
    os.mkdir('models/{}'.format(FOLDER))

# Create the data if it doesn't exist
if not os.path.exists('data/train/total_data.csv') or not os.path.exists('data/train/final_laptop_test_data.csv'):
    create_data()

# Get the data from the file
total_data = pd.read_csv('data/train/total_data.csv', index_col=False)
del total_data['index']

# Drop the Unnamed column
total_data = remove_misc(total_data)

# Convert the dataframe to numpy
total_data = total_data.to_numpy()
Common.M = total_data.shape[0]

# The split between training and test/validation 
split_size = 8000

train_data = total_data[:Common.M - split_size][:, 0:2]
print('Training shape: ' + str(train_data.shape))

val_data = total_data[Common.M - split_size: Common.M - (split_size//2)][:, 0:2]
print('Validation shape: ' + str(val_data.shape))

test_data = total_data[Common.M - (split_size//2):][:, 0:2]
print('Test shape: ' + str(test_data.shape))

train_labels = total_data[:Common.M - split_size][:, 2].astype('float32')
print('Training labels shape:', str(train_labels.shape))

val_labels = total_data[Common.M - split_size: Common.M - (split_size//2)][:, 2].astype('float32')
print('Val labels shape:', str(val_labels.shape))

test_labels = total_data[Common.M - (split_size//2):][:, 2].astype('float32')
print('Test labels shape:', str(test_labels.shape))

# Get the test laptop data
test_laptop_data = pd.read_csv('data/train/final_laptop_test_data.csv')
test_laptop_data = remove_misc(test_laptop_data).to_numpy()

# Split the data into the titles and the labels
test_laptop_labels = test_laptop_data[:, 2].astype('float32')
test_laptop_data = test_laptop_data[:, 0:2]
print('Laptop test shape:', str(test_laptop_data.shape))
print('Laptop test labels shape:', str(test_laptop_labels.shape))

# Initialize the model
net = None
if using_model == "characterbert":
    from src.model_architectures.characterbert_classifier import SiameseNetwork, forward_prop
    net = SiameseNetwork().to(Common.device)

elif using_model == "bert":
    from src.model_architectures.bert_classifier import SiameseNetwork, forward_prop
    net = SiameseNetwork(Common.MAX_LEN).to(Common.device)

elif using_model == "scaled characterbert concat":
    from src.model_architectures.characterbert_transformer_concat import SiameseNetwork, forward_prop
    net = SiameseNetwork(Common.MAX_LEN * 2 + 3)

elif using_model == "scaled characterbert add":
    from src.model_architectures.characterbert_transformer_add import SiameseNetwork, forward_prop
    net = SiameseNetwork().to(Common.device)

# Using cross-entropy because we are making a classifier
criterion = nn.CrossEntropyLoss()

# Using Adam optimizer
opt = AdamW(net.parameters(), lr=5e-5, weight_decay=0.001)
#opt = optim.Adam(net.parameters(), lr=1e-5)

print("************* TRAINING *************")

# 10 epochs
for epoch in range(10):
    # The size of each mini-batch
    BATCH_SIZE = 16

    # How long we should accumulate for running loss and accuracy
    PERIOD = 50
    
    # Iterate through each training batch
    net.train()
    current_batch = 0
    running_loss = 0.0
    running_accuracy = 0.0
    for i, position in enumerate(range(0, len(train_data), BATCH_SIZE)):
        current_batch += 1
        if (position + BATCH_SIZE > len(train_data)):
            batch_data = train_data[position:]
            batch_labels = train_labels[position:]
        else:
            batch_data = train_data[position:position + BATCH_SIZE]
            batch_labels = train_labels[position:position + BATCH_SIZE]
            
        # Zero the parameter gradients
        opt.zero_grad()
        
        # Forward propagation
        loss, accuracy = forward_prop(batch_data, batch_labels, net, criterion)

        # Add to both the running accuracy and running loss (every 10 batches)
        running_accuracy += accuracy
        running_loss += loss.item()

        # Backprop
        loss.backward()

        # Clip the gradient to minimize chance of exploding gradients
        torch.nn.utils.clip_grad_norm_(net.parameters(), 0.01)

        # Apply the gradients
        opt.step()
        
        # Print statistics every batch
        print('Training Epoch: %d, Batch %5d, Loss: %.6f, Accuracy: %.6f, Running Loss: %.6f, Running Accuracy %.6f' %
                (epoch + 1, i + 1, loss, accuracy, running_loss / current_batch, running_accuracy / current_batch))
        
        # Clear our running variables every 10 batches
        if (current_batch == PERIOD):
            current_batch = 0
            running_loss = 0
            running_accuracy = 0

    torch.save(net.state_dict(), 'models/{}/{}.pt'.format(FOLDER, MODEL_NAME + '_epoch' + str(epoch + 1)))

    # Iterate through each validation batch
    net.eval()
    running_loss = 0
    running_accuracy = 0
    current_batch = 0
    for i, position in enumerate(range(0, len(val_data), BATCH_SIZE)):
        current_batch += 1
        if (position + BATCH_SIZE > len(val_data)):
            batch_data = val_data[position:]
            batch_labels = val_labels[position:]
        else:
            batch_data = val_data[position:position + BATCH_SIZE]
            batch_labels = val_labels[position:position + BATCH_SIZE]
        
        # Forward propagation
        loss, accuracy = forward_prop(batch_data, batch_labels, net, criterion)

        # Add to running loss and accuracy (every 10 batches)
        running_accuracy += accuracy
        running_loss += loss.item()

        # Print statistics every batch
        print('Validation Epoch: %d, Batch %5d, Loss: %.6f, Accuracy: %.6f, Running Loss: %.6f, Running Accuracy: %.6f' %
                (epoch + 1, i + 1, loss, accuracy, running_loss / current_batch, running_accuracy / current_batch))

        # Clear our running variables every 10 batches
        if (current_batch == PERIOD):
            current_batch = 0
            running_loss = 0
            running_accuracy = 0

    # Iterate through the test laptop data
    running_loss = 0.0
    running_accuracy = 0.0
    current_batch = 0
    for i, position in enumerate(range(0, len(test_laptop_data), BATCH_SIZE)):
        current_batch += 1
        if (position + BATCH_SIZE > len(test_laptop_data)):
            batch_data = test_laptop_data[position:]
            batch_labels = test_laptop_labels[position:]
        else:
            batch_data = test_laptop_data[position:position + BATCH_SIZE]
            batch_labels = test_laptop_labels[position:position + BATCH_SIZE]

        # Forward propagation
        loss, accuracy = forward_prop(batch_data, batch_labels, net, criterion)

        # Add to running loss and accuracy (every 10 batches)
        running_loss += loss.item()
        running_accuracy += accuracy
        
        # Print statistics every batch
        print('Test Laptop Epoch: %d, Batch %5d, Loss: %.6f, Accuracy: %.6f, Running Loss: %.6f, Running Accuracy: %.6f' %
                (epoch + 1, i + 1, loss, accuracy, running_loss / current_batch, running_accuracy / current_batch))

        # Clear our running variables every 10 batches
        if (current_batch == PERIOD):
            current_batch = 0
            running_loss = 0
            running_accuracy = 0
