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
if not os.path.exists('data/train/total_data.csv'):
    create_data()

# Get the data from the file
total_data = pd.read_csv('data/train/total_data.csv')

# Drop the Unnamed column
total_data = remove_misc(total_data)

Common.MAX_LEN = get_max_len(total_data)

# Convert the dataframe to numpy
total_data = total_data.to_numpy()
SIZE = total_data.shape[0]

# The split between training and test/validation 
split_size = 10000

train_data = total_data[:SIZE - split_size][:, 0:2]
print('Training shape: ' + str(train_data.shape))

val_data = total_data[SIZE - split_size: SIZE - (split_size//2)][:, 0:2]
print('Validation shape: ' + str(val_data.shape))

test_data = total_data[SIZE - (split_size//2):][:, 0:2]
print('Test shape: ' + str(test_data.shape))

train_labels = total_data[:SIZE - split_size][:, 2].astype('float32')
print('Training labels shape:', str(train_labels.shape))

val_labels = total_data[SIZE - split_size: SIZE - (split_size//2)][:, 2].astype('float32')
print('Val shape:', str(val_labels.shape))

test_labels = total_data[SIZE - (split_size//2):][:, 2].astype('float32')
print('Test shape:', str(test_labels.shape))

# Initialize the model
net = SiameseNetwork(Common.MAX_LEN)

# Using cross-entropy because we are making a classifier
criterion = nn.CrossEntropyLoss()

# Using Adam optimizer
opt = AdamW(net.parameters(), lr=1e-6)

print("************* TRAINING *************")

# 5 epochs
for epoch in range(10):
    # The size of each mini-batch
    BATCH_SIZE = 32
    
    # Iterate through each batch
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
        accuracy = accuracy = torch.sum(torch.argmax(forward, dim=1) == batch_labels) / float(forward.size()[0])
        
        # Backprop
        loss.backward()
        
        # Apply the gradients
        opt.step()
        
        # Print statistics every batch
        print('Epoch: %d, Batch %5d, Loss: %.6f, Accuracy: %.3f' %
                (epoch + 1, i + 1, loss, accuracy))
