import tensorflow as tf
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Input, Dropout, Lambda
from tensorflow.keras.regularizers import l2
from src.common import Common
from src.model_architectures.model_functions import create_embeddings, exp_distance

def siamese_network(input_shape):
    # Defines our inputs
    left_title = Input(input_shape, dtype='string')
    right_title = Input(input_shape, dtype='string')
    
    # Create embeddings
    CreateEmbeddings = Lambda(create_embeddings, output_shape=(None, Common.MAX_LEN, Common.EMBEDDING_SHAPE[0]))
    left_embeddings = CreateEmbeddings(left_title)
    right_embeddings = CreateEmbeddings(right_title)
        
    # The LSTM units
    model = tf.keras.Sequential(name='siamese_model')
    model.add(Bidirectional(LSTM(units=128,
                                 name='lstm_1',
                                 return_sequences=True,
                                 activity_regularizer=l2(0.007),
                                 recurrent_regularizer=l2(0.0002), 
                                 kernel_regularizer=l2(0.0002))))
    model.add(Dropout(rate=0.5))
    model.add(Bidirectional(LSTM(units=64,
                             name='lstm_2',
                             return_sequences=True,
                             activity_regularizer=l2(0.007),
                             recurrent_regularizer=l2(0.0002), 
                             kernel_regularizer=l2(0.0002))))
    model.add(Dropout(rate=0.5))
    model.add(Bidirectional(LSTM(units=64,
                         name='lstm_3',
                         return_sequences=True,
                         activity_regularizer=l2(0.007),
                         recurrent_regularizer=l2(0.0002), 
                         kernel_regularizer=l2(0.0002))))
    model.add(Dropout(rate=0.5))
    model.add(Bidirectional(LSTM(units=64,
                             #return_sequences=True,
                             name='lstm_4',
                             activity_regularizer=l2(0.007),
                             recurrent_regularizer=l2(0.0002), 
                             kernel_regularizer=l2(0.0002))))
    model.add(Dropout(rate=0.5))
    # The dense layers
    model.add(Dense(units=512, activation='elu', name='dense_1'))
    model.add(Dropout(rate=0.6))
    model.add(Dense(units=256, activation='elu', name='dense_2'))
    
    # Forward propagate through the model to generate the encodings
    encoded_left_title = model(left_embeddings)
    encoded_right_title = model(right_embeddings)

    # Take the difference and then exponentiate it 
    Distance = Lambda(exp_distance)
    distance = Distance([encoded_left_title, encoded_right_title])

    # Send the distance to a dense layer
    distance = Dense(units=128, activation='elu', name='dense_3')(distance)

    # Send the dense layer to the sigmoid classifier
    distance = Dropout(0.5)(distance)
    prediction = Dense(units=1, activation='sigmoid', kernel_regularizer=l2(0.0001))(distance)
    
    # Create and return the network
    siamese_net = tf.keras.Model(inputs=[left_title, right_title], outputs=prediction, name='siamese_network')
    return siamese_net