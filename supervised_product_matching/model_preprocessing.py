import numpy as np
from nltk.corpus import stopwords
from transformers import AutoTokenizer
from character_bert.utils.character_cnn import CharacterIndexer
from supervised_product_matching.config import ModelConfig

# CharacterBERT tokenizer
character_indexer = CharacterIndexer()

# BERT tokenizer
bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def remove_stop_words(phrase, omit_punctuation=[]):
    '''
    Removes the stop words from a string
    '''

    # Creates the stopwords
    to_stop = stopwords.words('english')
    punctuation = "!”#$%&’()*+,-./:;<=>?@[\]^_`{|}~ "
    for x in omit_punctuation:
        if x in punctuation:
            punctuation = punctuation.replace(x, '')
    for c in punctuation:
        to_stop.append(c)
    to_stop.append('null')
    
    for punc in punctuation:
        phrase = phrase.replace(punc, ' ')
    
    return ' '.join((' '.join([x for x in phrase.split(' ') if x not in to_stop])).split()).lower()

def add_tags(arr):
    '''
    Append the [CLS] and [SEP] tags to a sequence
    '''
    
    return np.char.add(
        np.char.add(
            np.char.add(
                np.char.add(
                    np.array(['[CLS] ']), 
                    arr[:, 0]
                ), 
                np.array([' [SEP] '])
            ), 
            arr[:, 1]
        ), 
        np.array([' [SEP]'])
    )

def character_bert_preprocess_batch(x, pad=False,):
    """
    Preprocess a batch before it goes into the CharacterBERT model
    """
    x = x.astype('U')

    # BERT for title similarity works having the two sentences (sentence1, sentence2)
    # and ordering them in both combinations that they could be (sentence1 + sentence2)
    # and (sentence2 + sentence1). That is why we do np.flip() on x (the input sentences)
    # add_tags just adds the [CLS] and [SEP] tags to the strings
    input1 = add_tags(x)
    input2 = add_tags(np.flip(x, 1))


    # We need to split up each token in the title by the space
    # So, "intel core i7 7700k" becomes ["intel", "core", "i7", "7700k"]
    input1 = np.char.split(input1)
    input2 = np.char.split(input2)

    # Now, we feed the input into the CharacterBERT tokenizer, which converts each 
    if pad:
        input1 = character_indexer.as_padded_tensor(input1, maxlen=ModelConfig.max_len * 2 + 3)
        input2 = character_indexer.as_padded_tensor(input2, maxlen=ModelConfig.max_len * 2 + 3)
    else:
        input1 = character_indexer.as_padded_tensor(input1)
        input2 = character_indexer.as_padded_tensor(input2)

    # Send the data to the GPU
    input1 = input1.to(ModelConfig.device)
    input2 = input2.to(ModelConfig.device)

    return (input1, input2)

def bert_preprocess_batch(x):
    """
    Preprocess a batch before it goes into BERT
    """

    # BERT for title similarity works having the two sentences (sentence1, sentence2)
    # and ordering them in both combinations that they could be (sentence1 + sentence2)
    # and (sentence2 + sentence1). That is why we do np.flip() on x (the input sentences)
    input1 = bert_tokenizer(x.tolist(),
                            return_tensors='pt',
                            padding='max_length',
                            truncation=True,
                            max_length=ModelConfig.max_len)

    input2 = bert_tokenizer(np.flip(x, 1).tolist(),
                            return_tensors='pt',
                            padding='max_length',
                            truncation=True,
                            max_length=ModelConfig.max_len)

    # Send the data to the GPU
    input1 = input1.to(ModelConfig.device)
    input2 = input2.to(ModelConfig.device)

    return (input1, input2)