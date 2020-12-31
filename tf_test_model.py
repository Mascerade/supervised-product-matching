import numpy as np
from src.common import Common
from src.preprocessing import remove_stop_words

# ## Manual Testing ###############################################################
# Converts titles into embeddings arrays and allow the model to make a prediction #
###################################################################################


# Using distance sigmoid by default
model_choice = 'distance-sigmoid'
MAX_LEN = 44

if model_choice == 'distance-sigmoid':
    print('Using the distance sigmoid model.')
    from src.model_architectures.distance_sigmoid import siamese_network
    model = siamese_network((MAX_LEN))

elif model_choice == 'exp-distance-sigmoid':
    print('Using the exponential distance sigmoid model.')
    from src.model_architectures.exp_distance_sigmoid import siamese_network
    model = siamese_network((MAX_LEN))

elif model_choice == 'manhattan-distance':
    print('Using the manhattan distance model.')
    from src.model_architectures.manhattan_distance import siamese_network
    model = siamese_network((MAX_LEN))

else:
    print('Using the exponential distance softmax.')
    from src.model_architectures.exp_distance_softmax import siamese_network
    model = siamese_network((MAX_LEN))

model.summary()

# Load the model using the weights
model.load_weights('models/DistanceSigmoid_40epoch_84%_val.h5')

title_one = 'True Wireless Earbuds VANKYO X200 Bluetooth 5 0 Earbuds in Ear TWS Stereo Headphones Smart LED Display Charging Case IPX8 Waterproof 120H Playtime Built Mic Deep Bass Sports Work'
title_two = 'TOZO T10 Bluetooth 5 0 Wireless Earbuds Wireless Charging Case IPX8 Waterproof TWS Stereo Headphones Ear Built Mic Headset Premium Sound Deep Bass Sport Black'

title_one_arr = [' '] * MAX_LEN
title_two_arr = [' '] * MAX_LEN
title_one = remove_stop_words(title_one.lower())
title_two = remove_stop_words(title_two.lower())

for idx, x in enumerate(title_one.split(' ')):
    title_one_arr[idx] = x

for idx, x in enumerate(title_two.split(' ')):
    title_two_arr[idx] = x

title_one_arr = np.array(title_one_arr).reshape(1, MAX_LEN).astype('<U22')
title_two_arr = np.array(title_two_arr).reshape(1, MAX_LEN).astype('<U22')

print(model.predict([title_one_arr, title_two_arr]))
