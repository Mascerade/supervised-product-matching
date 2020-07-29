## Updates Regarding Network
* Update 6/29/2020: I have determined the way I will go about the embedding and the actual neural network
   * First, I will use fasttext to create the embeddings BEFORE inputting into the siamese network. This will make it so that we can use the n-grams function of fasttext.
   * The only downside of this is that the input to the network is the embedding instead of the text data which is probably looked down upon.
   * The acutal neural network is LSTM -> Dropout -> LSTM -> Dropout -> LSTM -> Dropout -> Dense -> Dropout -> Dense and the last dense layer is the output, which will will be used for the constrastive loss function

* Update 7/1/2020: This simply is not working. The training set loss gets stuck around 0.3330 with an accuracy around 0.80 and the validation loss is ~1.990 meaning it is clearly not generalizing well. I don't think constrastive loss is the way to go about this problem.

* Update 7/1/2020: I changed to a sigmoid output and binary cross-entropy as the loss function. Hopefully this will act as more of a similarity function
   * In addition, the dataset is messed up. There are about 10,000 positive examples and 40,000 negative examples, which explains why the model was simply learning to predict negative on everything, yielding an 80% accuracy on training, while getting 100% on the validation and 100% on the test. I will have to properly cleanse the data.

* Update 7/2/2020: I organized the data to have an equal balance of postive and negative examples. Now, there are 19,380 total examples of title pairs split 50-50 between positive and negative.
   * Training on this we are still just getting training accuracy equivalent to the distribution of data of the training set. Very frustrating. We used 2 epochs and a batch size of 64.
   * This is still using contrastive loss

* Update 7/2/200: FINALLYYYYYYYY!!!!! I got the model to work! Instead of using contrastive loss, I used a square-distance layer for the encodings and fed that into a softmax layer that outputs the probability of a match vs. not a match. For the loss function, I used Cross-Entropy Loss. I got an accuracy of 87% on the test set with ~91% on the training set, which means there is a bit of variance, only ~4%.  
