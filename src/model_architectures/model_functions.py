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

def exp_distance(vectors):
    x, y = vectors
    return tf.exp(-tf.subtract(x, y))

def manhattan_distance(vectors):
    x, y = vectors
    """
    Helper function for the similarity estimate of the LSTMs outputs
    """
    return tf.exp(-tf.reduce_sum(tf.abs(x - y), axis=1, keepdims=True))

def create_embeddings(vectors):
    out = tf.numpy_function(func=to_embeddings, inp=[vectors], Tout='float32')
    out.set_shape((None, Common.MAX_LEN, Common.EMBEDDING_SHAPE[0]))
    return out

def constrastive_loss(y_true, y_pred):
    """
    Note: for the constrastive loss, because 0 denotes that they are from the same class
    and one denotes they are from a different class, I swaped the (Y) and (1 - Y) terms
    """
    margin = 2.0
    d = y_pred
    d_sqrt = tf.sqrt(d)
    loss = (y_true * d) + ((1 - y_true) * tf.square(tf.maximum(0., margin - d_sqrt)))
    loss = 0.5 * tf.reduce_mean(loss)
    return loss

def constrastive_accuracy(y_true, y_pred):
    """
    Accuracy metric for constrastive loss because values close to 0 are equal and values high are different
    0.5 is the threshold here

    """
    return tf.reduce_mean(tf.cast(tf.equal(y_true, tf.cast(y_pred < 0.5, y_true.dtype)), y_true.dtype))

def save_model(model, name):
    """
    Saves a model with a particular name
    """
    model.save('models/' + name + '.h5')
