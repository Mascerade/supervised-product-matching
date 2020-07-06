import numpy as np
from src.model import siamese_network
from src.common import Common

# ## Manual Testing ###############################################################
# Converts titles into embeddings arrays and allow the model to make a prediction #
###################################################################################

# Get the model architecture
model = siamese_network((Common.MAX_LEN, Common.EMBEDDING_SHAPE[0],))
model.summary()

# Load the model using the weights
model.load_weights('models/Softmax-LSTM-50-epochs.h5')

title_one = 'True Wireless Earbuds VANKYO X200 Bluetooth 5 0 Earbuds in Ear TWS Stereo Headphones Smart LED Display Charging Case IPX8 Waterproof 120H Playtime Built Mic Deep Bass Sports Work'
title_two = 'TOZO T10 Bluetooth 5 0 Wireless Earbuds Wireless Charging Case IPX8 Waterproof TWS Stereo Headphones Ear Built Mic Headset Premium Sound Deep Bass Sport Black'
title_one_arr = np.zeros((1, 42, 300))
title_two_arr = np.zeros((1, 42, 300))
title_one.lower()
title_two.lower()
for idx, word in enumerate(title_one.split(' ')):
    title_one_arr[0, idx] = Common.fasttext_model[word]
    
for idx, word in enumerate(title_two.split(' ')):
    title_two_arr[0, idx] = Common.fasttext_model[word]

print(model.predict([title_one_arr, title_two_arr]))
