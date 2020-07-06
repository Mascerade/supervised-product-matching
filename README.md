# Product Matching Neural Network

## Usage
The iPython Notebook is meant to be used as a testing ground for the neural network. `TestingGrounds.ipynb` is just a place to understand the different functions and the models. It is also where I write new code before putting it into actual python files.

`train_model.py` is where to train the model. It is faster to do in the console as opposed to an iPython Notebook. 

`test_model.py` allows you to test your own different titles/strings to see how well the model does. It outputs a list where the first value is the probability the two titles do not represent the same entity and the second value is the probability that they do.

The  `src` directory are the helper functions to create data and generate the mode.

The `models` directory contains the different models trained so far and also the fastText model (read the important section to download it)

#### Important
* Download the [training set used](http://data.dws.informatik.uni-mannheim.de/largescaleproductcorpus/data/v2/trainsets/cameras_train.zip) and put it into the `computers_train` directory.

* Download and unzip the [fastText model](https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M-subword.zip) and put the `.bin` file into the `models` directory

## Notes About Network

### Fasttext
* I have just been experimenting with the FastText word embedding matrix
   * I got the largest one they had (600 billion tokens) and it works with n-grams so even if a word is not explicitly in the word embedding, it will stil find a similar word to it and get a word embedding based on that.

### My Network
* This model uses an embedding matrix using the FastText model, then I have two of the same networks (LSTM) for each title, making a siamese network, and then goes to two dense layers and a softmax at the end, which makes the prediction of whether the two titles are actually the same.
* Model: Embedding (FastText) -> LSTM -> Dropout -> LSTM -> Dropout -> LSTM -> Dropout-> Dense -> Dropout -> Dense -> Square Distance Between Encodings -> Softmax
   * \* The FastText embedding layer is not actually part of the network. The training data is just the titles converted to the FastText embeddings, and then you input FastText embeddings for predictions.

### Dataset
* For the training data, I am going to use the WDC Gold Standard database of products (http://webdatacommons.org/largescaleproductcorpus/v2/index.html)

* After further looking at the data, it seems like each product in the dataset as a cluser_id.
   * Essentially, this means that we can just use the cluster_id to build the positive examples of matches and use random examples outside of the cluster to build our negative training examples

* The way the dataset works is that for each entry in it, there is a title_right and a title_left. These are the two titles represented in a pair. There is a label that tells you whether the titles represent the same product or not. Using this, we already have built a dataset of positive and negative example pairs.
   * I also have created a seperate CSV file that is a simpler version of teh original dataset. It ony containes the titles with a label (either 0 or 1)

* The WDC Product Data researchers also did their own experiments, and they actually have a model that they trained using their own custom fasttext encoding model
   * They got about 90% accuracy on their training set which included computers, cameras, watches and shoes
      * They used a regular vanilla RNN which I find odd because an LSTM would surely capture the relatability between the tokens much better than a standard LSTM
   * I find this odd because computers and cameras do not make up a lot of the training set
      * There are 26 million offers, with office products making up about 13.13% of the dataset, so surely they could have used some of those
      * If the issue were the correlation between the different product categories (electronics related to book related to health related to toys etc.), then why would they pair computers and cameras with watches and shoes?

### Updates Regarding Network
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
