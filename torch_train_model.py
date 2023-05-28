import pandas as pd
import numpy as np
import requests
import os
import sys
import gc
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix

""" LOCAL IMPORTS """
from src.data_preprocessing import remove_misc
from src.common import Common
from create_data import create_data

# The size of each mini-batch
BATCH_SIZE = 4

# The size of the validation mini-batch
VAL_BATCH_SIZE = 2

# Data size for training
TRAIN_SIZE = 455000

# How long we should accumulate for running loss and accuracy
PERIOD = 50

def usage():
    print('Usage: torch_train_model.py [OPTIONS] <SUBCOMMAND> [ARGS]')
    print('  OPTIONS:')
    print('     -O <folder> <model-name>   The folder to output the models generated and the name they will use. Folder default is "default", model name default is "model"')
    print('     -M <model-to-use>          Give the name of the model to use for training. Options are bert, characterbert, scaled-characterbert-concat, scaled-charactertbert-add. Default is characterbert.')
    print('     -visualizer                Send data to NLP Dashboard to see training results in real-time.')
    print('     -dtable                    Delete the database for NLP Dashboard before creating new one (must come after -O option).')
    print('  SUBCOMMAND:')
    print('     --help                     Prints out this usage information and exit.')

def split_test_data(df):
    '''
    Split test data into the data and the labels
    '''

    df = remove_misc(df).to_numpy()
    df_labels = df[:, 2].astype('float32')
    df_data = df[:, 0:2]
    return df_data, df_labels

def send_batch_data(epoch, batch_num, batch_data, batch_size, forward, labels, accuracy, loss, running_accuracy, running_loss, table):
    # To send the training examples, we need the epoch and batch number on each example
    batch_epoch = np.tile(np.array([epoch, batch_num]), (batch_size, 1))

    # Includes, the epoch, batch number, positive softmax, negative softmax, prediction, and labels
    train_examples = np.concatenate((batch_epoch, # epoch/batch
                                    batch_data, # titles
                                    np.round(forward[:, 1].cpu().detach().numpy().reshape(batch_size, 1), 4).astype(str).astype(float),
                                    np.round(forward[:, 0].cpu().detach().numpy().reshape(batch_size, 1), 4).astype(str).astype(float),
                                    torch.argmax(forward, dim=1).cpu().detach().numpy().reshape(batch_size, 1),
                                    labels.astype(int).reshape(batch_size, 1)),
                                    axis=1)

    # Need to put the data into a dictionary to send it, so these are the keys for sending the batch data and training examples
    put_batch_labels = ['epoch', 'batch', 'accuracy', 'loss', 'runningAccuracy', 'runningLoss']
    train_examples_labels = ['epoch', 'batch', 'title1', 'title2', 'positivePercentage', 'negativePercentage', 'modelPrediction', 'label']
    train_examples = train_examples.tolist()

    # Going to use zip() to create the dictionary, so have a list of the elements in order
    batch_info = [epoch, 
                  batch_num,
                  float('%.4f'%(accuracy)),
                  float('%.4f'%(loss.item())),
                  float('%.4f'%(running_accuracy)),
                  float('%.4f'%(running_loss))]
    
    # Put the data that needs to be send into dictionaries
    batch_info = [dict(zip(put_batch_labels, batch_info))]
    train_examples_data = []
    for example in train_examples:
        train_examples_data.append(dict(zip(train_examples_labels, example)))
    train_examples_data = [train_examples_data]
    
    # Make the put request
    requests.put('http://localhost:3000/add_batch_data', json={'model_name': model_name, 'data': batch_info, 'table': table})
    requests.put('http://localhost:3000/add_examples_data', json={'model_name': model_name, 'data': train_examples_data, 'table': table})

def validation(net, epoch, data, labels, using_dashboard, name):
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

        try:
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

            # Send the data to the NLPDashboardServer
            if using_dashboard:   
                send_batch_data(epoch,
                                i + 1,
                                batch_data,
                                VAL_BATCH_SIZE,
                                forward,
                                batch_labels,
                                accuracy,
                                loss,
                                running_accuracy / current_batch,
                                running_loss / current_batch,
                                name)

            # Print statistics every batch
            #print("Torch memory allocator: {} bytes".format(torch.cuda.memory_reserved()))
            print('%s Batch: %5d, Loss: %.6f, Accuracy: %.6f, Running Loss: %.6f, Running Accuracy: %.6f, Precision: %.3f, Recall: %.3f, F1 Score: %.3f' %
                    (name, i + 1, loss, accuracy, running_loss / current_batch, running_accuracy / current_batch, precision, recall, f1_score))

            # Clear our running variables every 10 batches
            if (current_batch == PERIOD):
                current_batch = 0
                running_loss = 0
                running_accuracy = 0
            
        
        except RuntimeError as e:
            if "out of memory" in str(e):
                print("WARNING: Ran out of memory. Skipping Batch.")
                gc.collect()
                torch.cuda.empty_cache()
    
    # Get the statistics for the whole data
    final_precision = running_tp / (running_tp + running_fp)
    final_recall = running_tp / (running_tp + running_fn)
    final_f1_score = 2 * ((final_precision * final_recall) / (final_precision + final_recall))
    print('%s: Precision: %.3f, Recall: %.3f, F1 Score: %.3f' % (name, final_precision, final_recall, final_f1_score))

if __name__ == '__main__':
    argv = sys.argv[1:]
    using_model = "characterbert"

    # Get the folder name in models
    folder = 'default'

    # Get the model name from the terminal
    model_name = 'model'

    # Parse the options
    while len(argv) > 0:
        if argv[0] == '-visualizer':
            argv = argv[1:]
            using_dashboard = True

        elif argv[0] == '-O':
            argv = argv[1:]
            folder = argv[0]
            argv = argv[1:]
            model_name = argv[0]
            argv = argv[1:]
        
        elif argv[0] == '-M':
            argv = argv[1:]
            using_model = argv[0]
            argv = argv[1:]
        
        elif argv[0] == '-dtable':
            argv = argv[1:]
            requests.delete('http://localhost:3000/delete_db', json={'model_name': model_name})

        else:
            break

    # Parse the subcommands
    while len(argv) > 0:
        if argv[0] == '--help':
            argv = argv[1:]
            usage()
            exit(0)
        
        else:
            break

    if using_dashboard:
        # Make POST request to model server
        requests.post('http://localhost:3000/create_db', json={'model_name': model_name, 'tables': ['Training',
                                                                                                    'Validation',
                                                                                                    'Test Laptop (General)',
                                                                                                    'Test Laptop (Same Title) (Space)',
                                                                                                    'Test Laptop (Same Title) (No Space)',
                                                                                                    'Test Laptop (Different Title) (Space)',
                                                                                                    'Test Laptop (Different Title) (No Space)']})

    print('\nOutputing models to {} with base name {}\n'.format(folder, model_name))

    # Create the folder for the model if it doesn't already exist
    if not os.path.exists('models/{}'.format(folder)):
        os.mkdir('models/{}'.format(folder))

    # Create the data if it doesn't exist
    if not os.path.exists('data/train/total_data.csv') or not os.path.exists('data/test/final_laptop_test_data.csv'):
        create_data()

    # Load the data
    train_data = pd.read_csv('data/train/total_data.csv', nrows=TRAIN_SIZE, chunksize=BATCH_SIZE)
    val_data = pd.read_csv('data/train/total_data.csv', skiprows=TRAIN_SIZE, names=['title_one', 'title_two', 'label', 'index'])
    del val_data['index']
    val_data = val_data.to_numpy()
    val_labels = val_data[:, 2].astype('float32')
    val_data = val_data[:, 0:2]

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

    elif using_model == "scaled-characterbert-concat":
        from supervised_product_matching.model_architectures.characterbert_transformer_concat import SiameseNetwork, forward_prop
        net = SiameseNetwork().to(Common.device)

    elif using_model == "scaled-characterbert-add":
        from supervised_product_matching.model_architectures.characterbert_transformer_add import SiameseNetwork, forward_prop
        net = SiameseNetwork().to(Common.device)
    else:
        print('Model {} not found.').format(using_model)
        sys.exit(1)

    # Using cross-entropy because we are making a classifier
    criterion = nn.CrossEntropyLoss()

    # Using Adam optimizer
    #opt = AdamW(net.parameters(), lr=1e-5, weight_decay=0.001)
    opt = optim.Adam(net.parameters(), lr=1e-5)

    print("************* TRAINING *************")

    # 10 epochs
    for epoch in range(10):
        # Iterate through each training batch
        net.train()
        current_batch = 0
        running_loss = 0.0
        running_accuracy = 0.0
        for i, position in enumerate(range(0, TRAIN_SIZE, BATCH_SIZE)):
            current_batch += 1
            batch_data = next(train_data)
            del batch_data['index']
            batch_data = batch_data.to_numpy()
            batch_labels = batch_data[:, 2].astype('float32')
            batch_data = batch_data[:, 0:2]
            
            try:
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
                
                # Send the data to the NLPDashboardServer
                if using_dashboard:
                    send_batch_data(epoch + 1,
                                    i + 1,
                                    batch_data,
                                    BATCH_SIZE,
                                    forward,
                                    batch_labels,
                                    accuracy,
                                    loss,
                                    running_accuracy / current_batch,
                                    running_loss / current_batch,
                                    'Training')

                # Print statistics every batch
                #print("Torch memory allocator: {} bytes".format(torch.cuda.memory_reserved()))
                print('Training Epoch: %d, Batch %5d, Loss: %.6f, Accuracy: %.6f, Running Loss: %.6f, Running Accuracy %.6f' %
                        (epoch + 1, i + 1, loss, accuracy, running_loss / current_batch, running_accuracy / current_batch))
                
                # Clear our running variables every 10 batches
                if (current_batch == PERIOD):
                    current_batch = 0
                    running_loss = 0
                    running_accuracy = 0


            except RuntimeError as e:
                if "out of memory" in str(e):
                    print("WARNING: Ran out of memory. Skipping Batch.")
                    gc.collect()
                    torch.cuda.empty_cache()

        train_data = pd.read_csv('data/train/total_data.csv', nrows=TRAIN_SIZE, chunksize=BATCH_SIZE)
        torch.save(net.state_dict(), 'models/{}/{}.pt'.format(folder, model_name + '_epoch' + str(epoch + 1)))

        # Test the model
        net.eval()
        validation(net, epoch + 1, val_data, val_labels, using_dashboard, 'Validation')
        validation(net, epoch + 1, test_laptop_data, test_laptop_labels, using_dashboard, 'Test Laptop (General)')
        validation(net, epoch + 1, test_gb_space_data, test_gb_space_labels, using_dashboard, 'Test Laptop (Same Title) (Space)')
        validation(net, epoch + 1, test_gb_no_space_data, test_gb_no_space_labels, using_dashboard, 'Test Laptop (Same Title) (No Space)')
        validation(net, epoch + 1, test_retailer_gb_space_data, test_retailer_gb_space_labels, using_dashboard, 'Test Laptop (Different Title) (Space)')
        validation(net, epoch + 1, test_retailer_gb_no_space_data, test_retailer_gb_no_space_labels, using_dashboard, 'Test Laptop (Different Title) (No Space)')