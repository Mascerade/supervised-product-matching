import pandas as pd
import numpy as np
import nltk
import os
import sys
import tensorflow as tf

""" LOCAL IMPORTS """
from src.preprocessing import remove_misc
from src.embeddings import load_embeddings_and_labels, save_embeddings, create_embeddings
from src.model import siamese_network, save_model
from src.common import Common, get_max_len
from src.laptop_data_creation import create_laptop_data
from src.pcpartpicker_data_creation import create_pcpartpicker_data
from create_data import create_data

MODELS = ['distance sigmoid', 'exp distance', 'manhattan distance sigmoid', 'exp distance softmax']
model = None

if len(sys.argv) > 0:
    if sys.argv[0] in MODELS:
        model = sys.argv
    
    else:
        model = 'exp distance'
        print('Using the default model (exp distance).')

else :
    print('Using the default model (exp distance).')
    model = 'exp distance'

create_data()

# Get the data from the file
total_data = pd.read_csv('data/train/total_data_NEW.csv')
MAX_LEN = get_max_len(total_data)

# Drop the Unnamed column
total_data = remove_misc(total_data)

# Organize the data into seperate dataframes
train_data1 = []
train_data2 = []
labels = []
total_iloc = total_data.iloc()
for idx in range(len(total_data)):
    title_one_base = [' '] * MAX_LEN
    title_two_base = [' '] * MAX_LEN
    row = total_iloc[idx]
    
    for row_idx, x in enumerate(row.title_one.split(' ')):
        title_one_base[row_idx] = x
    
    for row_idx, x in enumerate(row.title_two.split(' ')):
        title_two_base[row_idx] = x
    
    train_data1.append(title_one_base)
    train_data2.append(title_two_base)
    labels.append(row.label)

train_data1 = np.asarray(train_data1)
train_data2 = np.asarray(train_data2)
labels = np.asarray(labels).astype(np.float32)

split_size = 10000
X_train1 = train_data1[:len(labels) - split_size]
X_train2 = train_data2[:len(labels) - split_size]
X_train = np.stack((X_train1, X_train2))
print('Training shape: ' + str(X_train.shape))

X_val1 = train_data1[len(labels) - split_size: len(labels) - (split_size//2)]
X_val2 = train_data2[len(labels) - split_size: len(labels) - (split_size//2)]
X_val = np.stack((X_val1, X_val2))
print('Val shape: ' + str(X_val.shape))


X_test1 = train_data1[len(labels) - (split_size//2):]
X_test2 = train_data2[len(labels) - (split_size//2):]
X_test = np.stack((X_test1, X_test2))
print('Test shape: ' + str(X_test.shape))

Y_train = labels[:len(labels) - split_size]
print('Training labels shape:', str(Y_train.shape))

Y_val = labels[len(labels) - split_size: len(labels) - (split_size//2)]
print('Val shape:', str(Y_val.shape))

Y_test = labels[len(labels) - (split_size//2):]
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

print('Creating the model graph . . . ')
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
