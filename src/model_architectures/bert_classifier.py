import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModel

class SiameseNetwork(nn.Module):
    def __init__(self, max_length, h_size=768):
        '''
        Model that uses BERT to classify the titles.
        max_length: The max length a title could be for padding purposes
        h_size: The hidden layer size for the classification token (CLS) in BERT (Default: 768)
        '''

        super(SiameseNetwork, self).__init__()
        self.h_size = h_size
        self.max_length = max_length * 2
        
        # BERT tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        
        # BERT model
        self.bert = AutoModel.from_pretrained("bert-base-uncased")
        
        # We want to freeze all parameters except the last couple for training
        for idx, param in enumerate(self.bert.parameters()):
            if idx < 85:
                param.requires_grad = False
        
        # Fully-Connected layers
        self.fc1 = nn.Linear(self.h_size, 384)
        self.fc2 = nn.Linear(384, 2)
        
        # Dropout for overfitting
        self.dropout = nn.Dropout(p=0.5)
        
        # Softmax for prediction
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        '''
        x is going to be a numpy array of [sentenceA, sentenceB].
        Model using BERT to make a prediction of whether the two titles represent 
        the same entity.
        '''

        # BERT for title similarity works having the two sentences (sentence1, sentence2)
        # and ordering them in both combinations that they could be (sentence1 + sentence2)
        # and (sentence2 + sentence1). That is why we do np.flip() on x (the input sentences)
        input1 = self.tokenizer(x.tolist(),
                                return_tensors='pt',
                                padding='max_length',
                                truncation=True,
                                max_length=self.max_length)

        input2 = self.tokenizer(np.flip(x, 1).tolist(),
                                return_tensors='pt',
                                padding='max_length',
                                truncation=True,
                                max_length=self.max_length)
        
        # Send the inputs through BERT
        # We index at 1 because that gives us the classification token (CLS)
        # that BERT talks about in the paper (as opposed to each hidden layer for each)
        # token embedding
        output1 = self.bert(**input1)[1]
        output2 = self.bert(**input2)[1]
        
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
