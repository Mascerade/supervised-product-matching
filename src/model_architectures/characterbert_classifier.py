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
from transformers import AutoModel
from characterbert_modeling.character_bert import CharacterBertModel

class SiameseNetwork(nn.Module):
    def __init__(self, h_size=768):
        '''
        Model that uses BERT to classify the titles.
        max_length: The max length a title could be for padding purposes
        h_size: The hidden layer size for the classification token (CLS) in BERT (Default: 768)
        '''

        super(SiameseNetwork, self).__init__()
        self.h_size = h_size
        
        # CharacterBERT model
        self.bert = CharacterBertModel.from_pretrained('./pretrained-models/general_character_bert/')

        # Fully-Connected layers
        self.fc1 = nn.Linear(self.h_size, 2)
        
        # Dropout for overfitting
        self.dropout_5 = nn.Dropout(p=0.5)

        # Dropout last
        self.dropout_7 = nn.Dropout(p=0.7)
        
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

    def forward(self, input1, input2):
        '''
        x is going to be a numpy array of [sentenceA, sentenceB].
        Model using CharacterBERT to make a prediction of whether the two titles represent 
        the same entity.
        '''

        # Send the inputs through BERT
        # We index at 1 because that gives us the classification token (CLS)
        # that BERT talks about in the paper (as opposed to each hidden layer for each)
        # token embedding
        output1 = self.bert(input1)[1]
        output2 = self.bert(input2)[1]

        # BERT calls for the addition of both 
        addition = output1 + output2

        # Dropout
        addition = self.dropout_5(addition)
        
        # Fully-Connected Layer 1 (input of 768 units and output of 384)
        addition = self.fc1(addition)
        
        # Dropout
        addition = self.dropout_7(addition)
        
        # Softmax Activation to get predictions
        addition = self.softmax(addition)
        
        return addition
