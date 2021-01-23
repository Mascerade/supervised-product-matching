# CharacterBERT citation for authors:
'''
Paper Title: CharacterBERT: Reconciling ELMo and BERT for Word-Level Open-Vocabulary Representations From Characters
Authors: Hicham El Boukkouri and Olivier Ferret and Thomas Lavergne and Hiroshi Noji and Pierre Zweigenbaum and Junichi Tsujii
The characterbert_modeling and characterbert_utils were also created by them
Their GitHub Repo is at: https://github.com/helboukkouri/character-bert
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModel
from characterbert_modeling.character_bert import CharacterBertModel
from characterbert_utils.character_cnn import CharacterIndexer

class SiameseNetwork(nn.Module):
    def __init__(self, h_size=768):
        '''
        Model that uses BERT to classify the titles.
        max_length: The max length a title could be for padding purposes
        h_size: The hidden layer size for the classification token (CLS) in BERT (Default: 768)
        '''

        super(SiameseNetwork, self).__init__()
        self.h_size = h_size
        
        # CharacterBERT tokenizer
        self.character_indexer = CharacterIndexer()
        
        # CharacterBERT model
        self.bert = CharacterBertModel.from_pretrained('./pretrained-models/general_character_bert/')

        # We want to freeze all parameters except the last couple for training
        # for idx, param in enumerate(self.bert.parameters()):
        #     if idx < 85:
        #         param.requires_grad = False

        # Fully-Connected layers
        self.fc1 = nn.Linear(self.h_size, 384)
        self.fc2 = nn.Linear(384, 2)
        
        # Dropout for overfitting
        self.dropout = nn.Dropout(p=0.6)
        
        # Softmax for prediction
        self.softmax = nn.Softmax(dim=1)

    def add_tags(self, arr):
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

    def forward(self, x):
        '''
        x is going to be a numpy array of [sentenceA, sentenceB].
        Model using CharacterBERT to make a prediction of whether the two titles represent 
        the same entity.
        '''

        # The batch is originally of type 'object' and for add_tags to work properly
        # they need to be unicode
        x = x.astype('U')

        # BERT for title similarity works having the two sentences (sentence1, sentence2)
        # and ordering them in both combinations that they could be (sentence1 + sentence2)
        # and (sentence2 + sentence1). That is why we do np.flip() on x (the input sentences)
        # add_tags just adds the [CLS] and [SEP] tags to the strings
        input1 = self.add_tags(x)
        input2 = self.add_tags(np.flip(x, 1))


        # We need to split up each token in the title by the space
        # So, "intel core i7 7700k" becomes ["intel", "core", "i7", "7700k"]
        input1 = np.char.split(input1)
        input2 = np.char.split(input2)

        # Now, we feed the input into the CharacterBERT tokenizer, which converts each 
        input1 = self.character_indexer.as_padded_tensor(input1)
        input2 = self.character_indexer.as_padded_tensor(input2)

        # Send the inputs through BERT
        # We index at 1 because that gives us the classification token (CLS)
        # that BERT talks about in the paper (as opposed to each hidden layer for each)
        # token embedding
        output1 = self.bert(input1)[1]
        output2 = self.bert(input2)[1]
        
        # BERT calls for the addition of both 
        addition = output1 + output2
        
        # Fully-Connected Layer 1 (input of 768 units and output of 384)
        addition = self.fc1(addition)
        
        # ReLU Activation
        addition = F.relu(addition)
        
        # Dropout
        addition = self.dropout(addition)
        
        # Fully-Connected Layer 2 (input of 384 units, out of 2 for Softmax)
        addition = self.fc2(addition)
        
        # Dropout
        addition = self.dropout(addition)
        
        # Softmax Activation to get predictions
        addition = self.softmax(addition)
        
        return addition
