import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout, Lambda, Concatenate

# ## Model Info ############################################################################################################################
#                                                                                                                                          #
# For the model, we are going to use LSTMs with a Softmax unit                                                                             #
# that will also be used to predict whether the two products are the same                                                                  #
#                                                                                                                                          #
# First, we have to convert the titles to embeddings through FastText before feeding into the LSTM.                                        #
# The embedding part of this model will not be a layer because:                                                                            #
# * The fasttext model would be time consuming and annoying to get to work with an embedding layer in Keras                                #
# * The fasttext model is not going to be getting its embeddings optimized, so there is really no point in adding it as an embedding layer #
############################################################################################################################################

def square_distance(vectors):
    x, y = vectors
    return tf.square(x - y)

def euclidean_dist_out_shape(shapes):
    # Both inputs are fed in, so just use one of them and get the first value in the shape
    shape1, _ = shapes
    return (shape1[0],)

def siamese_network(input_shape):
    # Defines our inputs
    left_title = Input(input_shape, dtype='float32')
    right_title = Input(input_shape, dtype='float32')
    
    # The LSTM units
    model = tf.keras.Sequential(name='siamese_model')
    model.add(LSTM(units=256, return_sequences=True, name='lstm_1'))
    model.add(Dropout(rate=0.5))
    model.add(LSTM(units=128, return_sequences=True, name='lstm_2'))
    model.add(Dropout(rate=0.5))
    model.add(LSTM(units=128, name='lstm_3'))
    model.add(Dropout(rate=0.5))
    
    # The dense layers
    model.add(Dense(units=1024, activation='elu', name='dense_1'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(units=512, activation='elu', name='dense_2'))
    
    # Forward propagate through the model to generate the encodings
    encoded_left_title = model(left_title)
    encoded_right_title = model(right_title)

    SquareDistanceLayer = Lambda(square_distance)
    distance = SquareDistanceLayer([encoded_left_title, encoded_right_title])
    
    prediction = Dense(units=2, activation='softmax')(distance)

    # Create and return the network
    siamese_net = tf.keras.Model(inputs=[left_title, right_title], outputs=prediction, name='siamese_network')
    return siamese_net

# Note: for the constrastive loss, because 0 denotes that they are from the same class
# and one denotes they are from a different class, I swaped the (Y) and (1 - Y) terms

def constrastive_loss(y_true, y_pred):
    """
    Note: for the constrastive loss, because 0 denotes that they are from the same class
    and one denotes they are from a different class, I swaped the (Y) and (1 - Y) terms
    """
    margin = 2.0
    d = y_pred
    d_sqrt = tf.sqrt(d)
    #tf.print('\nY Pred: ', d, 'Shape: ', tf.shape(d))
    #tf.print('\nY True: ', y_true, 'Shape: ', tf.shape(y_true))
    
    loss = (y_true * d) + ((1 - y_true) * tf.square(tf.maximum(0., margin - d_sqrt)))
    
    #tf.print('\n Constrastive Loss: ', loss, 'Shape: ', tf.shape(loss))
    loss = 0.5 * tf.reduce_mean(loss)
    
    return loss

# Accuracy metric for constrastive loss because values close to 0 are equal and values high are different
# 0.5 is the threshold here
def constrastive_accuracy(y_true, y_pred):
    return tf.reduce_mean(tf.cast(tf.equal(y_true, tf.cast(y_pred < 0.5, y_true.dtype)), y_true.dtype))

def save_model(model, name):
    """
    Saves a model with a particular name
    """
    model.save('models/' + name + '.h5')
