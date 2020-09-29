import pandas as pd
import numpy as np
import nltk
import os
import sys
import tensorflow as tf

""" LOCAL IMPORTS """
from src.preprocessing import remove_misc
from src.embeddings import load_embeddings_and_labels, save_embeddings, create_embeddings
from src.common import Common, get_max_len
from create_data import create_data
from src.model_architectures.model_functions import save_model

MODELS = ['distance-sigmoid', 'exp-distance-sigmoid', 'manhattan-distance', 'exp-distance-softmax']
model_choice = None
model_name = None
print(sys.argv)
if len(sys.argv) >= 3:
    if sys.argv[1] in MODELS:
        model_choice = sys.argv[1]
    
    else:
        print('Model not found . . . ')
        sys.exit()
    
    model_name = sys.argv[2]

else :
    print('Provide a MODEL ARCHITECTURE and NAME for the model . . . ')
    sys.exit()

print(model_choice, model_name)

# Create the data if it doesn't exist
if not os.path.exists('data/train/total_data.csv'):
    create_data()

# Convert floats to one-hot arrays
def convert_to_one_hot(Y, C):
    """
    Function to create the 
    """
    Y = np.eye(C)[Y.reshape(-1)]
    return Y

# Get the data from the file
total_data = pd.read_csv('data/train/total_data.csv')
Common.MAX_LEN = get_max_len(total_data)

# Drop the Unnamed column
total_data = remove_misc(total_data)

# Organize the data into seperate dataframes
train_data1 = []
train_data2 = []
labels = []
total_iloc = total_data.iloc()
for idx in range(len(total_data)):
    title_one_base = [' '] * Common.MAX_LEN
    title_two_base = [' '] * Common.MAX_LEN
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

# The split between training and test/validation 
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

# Only convert to one-hot if we are using a softmax model
if 'softmax' in model_choice:
    Y_train = convert_to_one_hot(Y_train.astype(np.int32), 2)
    Y_val = convert_to_one_hot(Y_val.astype(np.int32), 2)
    Y_test = convert_to_one_hot(Y_test.astype(np.int32), 2)


# Choosing the architecture based on the argument
model = None
if model_choice == 'distance-sigmoid':
    print("Using the distance sigmoid model.")
    from src.model_architectures.distance_sigmoid import siamese_network
    model = siamese_network((Common.MAX_LEN))

elif model_choice == 'exp-distance-sigmoid':
    print("Using the exponential distance sigmoid model.")
    from src.model_architectures.exp_distance_sigmoid import siamese_network
    model = siamese_network((Common.MAX_LEN))

elif model_choice == 'manhattan-distance':
    print("Using the manhattan distance model.")
    from src.model_architectures.manhattan_distance import siamese_network
    model = siamese_network((Common.MAX_LEN))

else:
    print("Using the exponential distance softmax.")
    from src.model_architectures.exp_distance_softmax import siamese_network
    model = siamese_network((Common.MAX_LEN))

print('Creating the model graph . . . ')
model.summary()

# Compile the model
print('Compiling the model...')
model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])

# Train the model
print('About to train...')
model.fit(x=[X_train1, X_train2], y=Y_train, batch_size=128, epochs=50, validation_data=([X_val[0], X_val[1]], Y_val))

# Test the model
results = model.evaluate([X_test1, X_test2], Y_test, batch_size=16)
print('test loss, test accuracy: ', results)

# Save the model
save_model(model_choice, model_name)
