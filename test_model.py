import pandas as pd
import numpy as np
import sys
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
import time

""" LOCAL IMPORTS """
from src.data_preprocessing import remove_misc
from supervised_product_matching.model_preprocessing import remove_stop_words, character_bert_preprocess_batch, bert_preprocess_batch
from src.common import Common

using_model = "characterbert"

# Get the folder name in models
FOLDER = sys.argv[1]

# Get the model name from the terminal
MODEL_NAME = sys.argv[2]

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
    from supervised_product_matching.model_architectures.characterbert_classifier import SiameseNetwork, forward_prop
    net = SiameseNetwork().to(Common.device)

elif using_model == "bert":
    from supervised_product_matching.model_architectures.bert_classifier import SiameseNetwork, forward_prop
    net = SiameseNetwork().to(Common.device)

elif using_model == "scaled characterbert concat":
    from supervised_product_matching.model_architectures.characterbert_transformer_concat import SiameseNetwork, forward_prop
    net = SiameseNetwork()

elif using_model == "scaled characterbert add":
    from supervised_product_matching.model_architectures.characterbert_transformer_add import SiameseNetwork, forward_prop
    net = SiameseNetwork().to(Common.device)

if (torch.cuda.is_available()):
    net.load_state_dict(torch.load('./models/{}/{}.pt'.format(FOLDER, MODEL_NAME)))
else:
    net.load_state_dict(torch.load('./models/{}/{}.pt'.format(FOLDER, MODEL_NAME), map_location=torch.device('cpu')))


# Using cross-entropy because we are making a classifier
criterion = nn.CrossEntropyLoss()

print("************* Validating *************")

# The size of each mini-batch
BATCH_SIZE = 32

# The size of the validation mini-batch
VAL_BATCH_SIZE = 16

# How long we should accumulate for running loss and accuracy
PERIOD = 50

def validation(data, labels, name):
    '''
    Validate the model
    '''

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

def inference():
    '''
    Test model using your own titles
    '''
    
    title1 = input('First title: ')
    title2 = input('Second title: ')
    
    title1 = remove_stop_words(title1)
    title2 = remove_stop_words(title2)
    
    data = np.array([title1, title2]).reshape(1, 2)
    forward = net(*character_bert_preprocess_batch(data))
    np_forward = forward.detach().numpy()[0]
    
    print('Output: {}'.format(torch.argmax(forward)))
    print('Softmax: Negative {:.4f}%, Positive {:.4f}%'.format(np_forward[0], np_forward[1]))

user_input = input('Would you like to validate, or manually test the model? (validate/test) ')

net.eval()
if user_input.lower() == 'validate':
    validation(test_laptop_data, test_laptop_labels, 'Test Laptop (General)')
    validation(test_gb_space_data, test_gb_space_labels, 'Test Laptop (Same Title) (Space)')
    validation(test_gb_no_space_data, test_gb_no_space_labels, 'Test Laptop (Same Title) (No Space')
    validation(test_retailer_gb_space_data, test_retailer_gb_space_labels, 'Test Laptop (Different Title) (Space)')
    validation(test_retailer_gb_no_space_data, test_retailer_gb_no_space_labels, 'Test Laptop (Different Title) (No Space)')

else:
    while True:
        inference()
