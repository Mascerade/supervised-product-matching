import pandas as pd
import numpy as np
import nltk
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModel, AdamW

""" LOCAL IMPORTS """
from src.preprocessing import remove_misc
from src.embeddings import load_embeddings_and_labels, save_embeddings, create_embeddings
from src.common import Common, get_max_len
from create_data import create_data
from src.model_architectures.model_functions import save_model
from src.model_architectures.bert_classifier import SiameseNetwork

# Create the data if it doesn't exist
if not os.path.exists('data/train/total_data.csv') or not os.path.exists('data/train/final_laptop_test_data.csv'):
    create_data()

# Get the data from the file
total_data = pd.read_csv('data/train/total_data.csv')

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
net = SiameseNetwork(Common.MAX_LEN)

# Using cross-entropy because we are making a classifier
criterion = nn.CrossEntropyLoss()

# Using Adam optimizer
opt = AdamW(net.parameters(), lr=1e-6)

print("************* TRAINING *************")

# 10 epochs
for epoch in range(10):
    # The size of each mini-batch
    BATCH_SIZE = 32
    
    # Iterate through each training batch
    net.train()
    for i, position in enumerate(range(0, len(train_data), BATCH_SIZE)):
        if (position + BATCH_SIZE > len(train_data)):
            batch_data = train_data[position:]
            batch_labels = train_labels[position:]
        else:
            batch_data = train_data[position:position + BATCH_SIZE]
            batch_labels = train_labels[position:position + BATCH_SIZE]
            
        # Zero the parameter gradients
        opt.zero_grad()
        
        # Forward propagation
        forward = net(batch_data)

        # Convert batch labels to Tensor
        batch_labels = torch.from_numpy(batch_labels).view(-1).long()

        # Calculate loss
        loss = criterion(forward, batch_labels)

        # Calculate accuracy
        accuracy = torch.sum(torch.argmax(forward, dim=1) == batch_labels) / float(forward.size()[0])
        
        # Backprop
        loss.backward()
        
        # Apply the gradients
        opt.step()
        
        # Print statistics every batch
        print('Training Epoch: %d, Batch %5d, Loss: %.6f, Accuracy: %.3f' %
                (epoch + 1, i + 1, loss, accuracy))

    # Iterate through each validation batch
    net.eval()
    for i, position in enumerate(range(0, len(val_data), BATCH_SIZE)):
        if (position + BATCH_SIZE > len(val_data)):
            batch_data = val_data[position:]
            batch_labels = val_labels[position:]
        else:
            batch_data = val_data[position:position + BATCH_SIZE]
            batch_labels = val_labels[position:position + BATCH_SIZE]


        # Zero the parameter gradients
        opt.zero_grad()
        
        # Forward propagation
        forward = net(batch_data)

        # Convert batch labels to Tensor
        batch_labels = torch.from_numpy(batch_labels).view(-1).long()

        # Calculate loss
        loss = criterion(forward, batch_labels)

        # Calculate accuracy
        accuracy = torch.sum(torch.argmax(forward, dim=1) == batch_labels) / float(forward.size()[0])
        
        # Print statistics every batch
        print('Validation Epoch: %d, Batch %5d, Loss: %.6f, Accuracy: %.3f' %
                (epoch + 1, i + 1, loss, accuracy))

    # Iterate through the test laptop data
    for i, position in enumerate(range(0, len(test_laptop_data), BATCH_SIZE)):
        if (position + BATCH_SIZE > len(test_laptop_data)):
            batch_data = test_laptop_data[position:]
            batch_labels = test_laptop_labels[position:]
        else:
            batch_data = test_laptop_data[position:position + BATCH_SIZE]
            batch_labels = test_laptop_labels[position:position + BATCH_SIZE]


        # Zero the parameter gradients
        opt.zero_grad()
        
        # Forward propagation
        forward = net(batch_data)

        # Convert batch labels to Tensor
        batch_labels = torch.from_numpy(batch_labels).view(-1).long()

        # Calculate loss
        loss = criterion(forward, batch_labels)

        # Calculate accuracy
        accuracy = torch.sum(torch.argmax(forward, dim=1) == batch_labels) / float(forward.size()[0])
        
        # Print statistics every batch
        print('Test Laptop Epoch: %d, Batch %5d, Loss: %.6f, Accuracy: %.3f' %
                (epoch + 1, i + 1, loss, accuracy))
