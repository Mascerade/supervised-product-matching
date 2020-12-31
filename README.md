# Product Matching Neural Network

## Usage
The iPython Notebook is meant to be used as a testing ground for the neural network. `TestingGrounds.ipynb` is just a place to understand the different functions and the models. It is also where I write new code before putting it into actual python files.

`torch_train_model.py` is where to train the model.

`TestingModels.ipnb.py` allows you to test your own different titles/strings to see how well the model does. 

The `src` directory are the functions that create data and generate the model.

The `models` directory contains the different models trained so far and also the fastText model (if you want to use the ).

The `data_scrapers` directory contains scripts to scrape data for creating training data.


#### For Old Models using FastText Embeddings
* Download and unzip the [fastText model](https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M-subword.zip) and put the `.bin` file into the `models` directory
   * <b>This is only if you want to use the old models</b>
   * Also must un-comment the loading of the fastText model in `src/common.py`

## Notes About The Network

### fastText Embeddings (OLD METHOD)
* I have just been experimenting with the fastText word embedding matrix
   * I got the largest one they had (600 billion tokens) and it works with n-grams so even if a word is not explicitly in the word embedding, it will stil find a similar word to it and get a word embedding based on that.

### LSTM + fastText Model (OLD METHOD)
* There are many variations of this model that I use. The architectures can be found in  `src/model_architectures`

### BERT (New)
* For the latest models (>= 0.2.0), I use a pre-trained BERT model ([Google's paper](https://arxiv.org/pdf/1810.04805.pdf))
* I simply fine-tune BERT on the new data (so the last couple layers of BERT) and add a classification head on top of it.
* The architecture can be found in `src/model_architectures/bert_classifier.py`

### Dataset

#### WDC Product Corpus
* For the training data, I am going to use the WDC Gold Standard database of products (http://webdatacommons.org/largescaleproductcorpus/v2/index.html)

* After further looking at the data, it seems like each product in the dataset as a cluser_id.
   * Essentially, this means that we can just use the cluster_id to build the positive examples of matches and use random examples outside of the cluster to build our negative training examples

* The way the dataset works is that for each entry in it, there is a title_right and a title_left. These are the two titles represented in a pair. There is a label that tells you whether the titles represent the same product or not. Using this, we already have built a dataset of positive and negative example pairs.
   * I also have created a seperate CSV file that is a simpler version of teh original dataset. It ony containes the titles with a label (either 0 or 1)

* The WDC Product Data researchers also did their own experiments, and they actually have a model that they trained using their own custom fastText encoding model
   * They got about 90% accuracy on their training set which included computers, cameras, watches and shoes
      * They used a regular vanilla RNN which I find odd because an LSTM would surely capture the relatability between the tokens much better than a standard LSTM
   * I find this odd because computers and cameras do not make up a lot of the training set
      * There are 26 million offers, with office products making up about 13.13% of the dataset, so surely they could have used some of those
      * If the issue were the correlation between the different product categories (electronics related to book related to health related to toys etc.), then why would they pair computers and cameras with watches and shoes?

#### Custom Made Laptop Data (laptops.csv)
* After creating the initial model (v0.1), I decided that I wanted to create data catered specifically to getting better at laptop data, as it is a common electronic to shop for

* The data can be found at https://www.kaggle.com/ionaskel/laptop-prices

* I normalize the data and then substitute different key attributes (CPU, Graphics, Size, etc.) in order to create negative data and remove certain attributes (Screentype, brand, laptop type, etc.) and add in random words manufacturers like to use, like "premium", "NEW", etc. in order to to create the positive data

* I also randomize the order of tokens so that the does not overfit to certain positions of tokens

### More Custom Made Laptop Data (spec_data.csv)
* Using 