# Product Matching Neural Network

## Usage
The iPython Notebook is meant to be used as a testing ground for the neural network. `TestingGrounds.ipynb` is just a place to understand the different functions and the models. It is also where I write new code before putting it into actual python files.

`train_model.py` is where to train the model. It is faster to do in the console as opposed to an iPython Notebook. 

`test_model.py` allows you to test your own different titles/strings to see how well the model does. It outputs a list where the first value is the probability the two titles do not represent the same entity and the second value is the probability that they do.

The  `src` directory are the helper functions to create data and generate the mode.

The `models` directory contains the different models trained so far and also the fastText model (read the important section to download it).

The `data_scrapers` directory contains scripts to scrape data for creating training data.


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

#### WDC Product Corpus
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

#### Custom Made Laptop Data
* After creating the initial model (v0.1), I decided that I wanted to create data catered specifically to getting better at laptop data, as it is a common electronic to shop for

* The data can be found at https://www.kaggle.com/ionaskel/laptop-prices

* I normalize the data and then substitute different key attributes (CPU, Graphics, Size, etc.) in order to create negative data and remove certain attributes (Screentype, brand, laptop type, etc.) and add in random words manufacturers like to use, like "premium", "NEW", etc. in order to to create the positive data
   * *You can see all the code for that in `TestingGrounds` (it has not been implemented for a release yet)