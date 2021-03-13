import pandas as pd
import numpy as np
import nltk
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModel, AdamW
from sklearn.metrics import confusion_matrix
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
if not os.path.exists('data/train/total_data.csv') or not os.path.exists('data/test/final_laptop_test_data.csv'):
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

def split_test_data(df):
    '''
    Split test data into the data and the labels
    '''

    df = remove_misc(df).to_numpy()
    df_labels = df[:, 2].astype('float32')
    df_data = df[:, 0:2]
    return df_data, df_labels

test_laptop_data, test_laptop_labels = split_test_data(pd.read_csv('data/test/final_laptop_test_data.csv')) # General laptop test data
test_gb_space_data, test_gb_space_labels = split_test_data(pd.read_csv('data/test/final_gb_space_laptop_test.csv')) # Same titles; Substituted storage attributes
test_gb_no_space_data, test_gb_no_space_labels = split_test_data(pd.read_csv('data/test/final_gb_no_space_laptop_test.csv')) # Same titles; Substituted storage attributes
test_retailer_gb_space_data, test_retailer_gb_space_labels = split_test_data(pd.read_csv('data/test/final_retailer_gb_space_test.csv')) # Different titles; Substituted storage attributes
test_retailer_gb_no_space_data, test_retailer_gb_no_space_labels = split_test_data(pd.read_csv('data/test/final_retailer_gb_no_space_test.csv')) # Different titles; Substituted storage attributes
print('Loaded all test files')

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
#opt = AdamW(net.parameters(), lr=5e-5, weight_decay=0.001)
opt = optim.Adam(net.parameters(), lr=1e-5)

print("************* TRAINING *************")

# The size of each mini-batch
BATCH_SIZE = 32

# The size of the validation mini-batch
VAL_BATCH_SIZE = 16

# How long we should accumulate for running loss and accuracy
PERIOD = 50

def validation(data, labels, name):
    running_loss = 0.0
    running_accuracy = 0.0
    current_batch = 0
    running_tn = 0
    running_fp = 0
    running_fn = 0
    running_tp = 0
    for i, position in enumerate(range(0, len(data), VAL_BATCH_SIZE)):
        current_batch += 1
        if (position + VAL_BATCH_SIZE > len(data)):
            batch_data = data[position:]
            batch_labels = labels[position:]
        else:
            batch_data = data[position:position + VAL_BATCH_SIZE]
            batch_labels = labels[position:position + VAL_BATCH_SIZE]

        # Forward propagation
        loss, forward = forward_prop(batch_data, batch_labels, net, criterion)
        
        # Get the predictions from the net
        y_pred = torch.argmax(forward, dim=1).cpu()

        # Calculate accuracy
        accuracy = np.sum(y_pred.detach().numpy() == batch_labels) / float(batch_labels.shape[0])

        # Get the confusion matrix and calculate precision, recall and F1 score
        confusion = confusion_matrix(batch_labels, y_pred.detach().numpy(), labels=[0, 1])
        tn, fp, fn, tp = confusion.ravel()
        running_tn += tn
        running_fp += fp
        running_fn += fn
        running_tp += tp
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1_score = 2 * ((precision * recall) / (precision + recall))

        # Add to running loss and accuracy (every 10 batches)
        running_loss += loss.item()
        running_accuracy += accuracy
        
        # Print statistics every batch
        print('%s Batch: %5d, Loss: %.6f, Accuracy: %.6f, Running Loss: %.6f, Running Accuracy: %.6f, Precision: %.3f, Recall: %.3f, F1 Score: %.3f' %
                (name, i + 1, loss, accuracy, running_loss / current_batch, running_accuracy / current_batch, precision, recall, f1_score))

        # Clear our running variables every 10 batches
        if (current_batch == PERIOD):
            current_batch = 0
            running_loss = 0
            running_accuracy = 0
    
    # Get the statistics for the whole data
    final_precision = running_tp / (running_tp + running_fp)
    final_recall = running_tp / (running_tp + running_fn)
    final_f1_score = 2 * ((final_precision * final_recall) / (final_precision + final_recall))
    print('%s: Precision: %.3f, Recall: %.3f, F1 Score: %.3f' % (name, final_precision, final_recall, final_f1_score))

# 7 epochs
for epoch in range(7):    
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
        loss, forward = forward_prop(batch_data, batch_labels, net, criterion)

        # Calculate accuracy
        accuracy = np.sum(torch.argmax(forward, dim=1).cpu().detach().numpy() == batch_labels) / float(forward.size()[0])

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

    # Test the model
    net.eval()
    validation(val_data, val_labels, 'Validation')
    validation(test_laptop_data, test_laptop_labels, 'Test Laptop (General)')
    validation(test_gb_space_data, test_gb_space_labels, 'Test Laptop (Same Title) (Space)')
    validation(test_gb_no_space_data, test_gb_no_space_labels, 'Test Laptop (Same Title) (No Space')
    validation(test_retailer_gb_space_data, test_retailer_gb_space_labels, 'Test Laptop (Different Title) (Space)')
    validation(test_retailer_gb_no_space_data, test_retailer_gb_no_space_labels, 'Test Laptop (Different Title) (No Space)')
