import tensorflow as tf
import numpy as np
from src.common import Common

def to_embeddings(data):
    embeddings = []
    for row in data:
        embeddings.append(np.array([Common.fasttext_model[str(x.decode('utf-8'))] for x in row]))
    
    embeddings = np.array(embeddings)
    return(embeddings)

def l1_distance(vectors):
    x, y = vectors
    return tf.abs(x - y)

def distance(vectors):
    x, y = vectors
    return tf.subtract(x, y)

def l2_distance(vectors):
    x, y = vectors
    return tf.square(x - y)

def cosine_similarity(vectors):
    x, y = vectors
    return tf.reduce_sum((tf.multiply(x, y) / tf.multiply(np.sqrt(tf.reduce_sum(tf.square(x))), np.sqrt(tf.reduce_sum(tf.square(y))))))

def manhattan_distance(vectors):
    x, y = vectors
    ''' Helper function for the similarity estimate of the LSTMs outputs'''
    return tf.exp(-tf.reduce_sum(tf.abs(x - y), axis=1, keepdims=True))

def create_embeddings(vectors):
    out = tf.numpy_function(func=to_embeddings, inp=[vectors], Tout='float32')
    out.set_shape((None, Common.MAX_LEN, Common.EMBEDDING_SHAPE[0]))
    return out

def euclidean_dist_out_shape(shapes):
    # Both inputs are fed in, so just use one of them and get the first value in the shape
    shape1, shape2 = shapes
    return (shape1[0],)
