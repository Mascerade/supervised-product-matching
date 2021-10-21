import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModel
from supervised_product_matching.config import ModelConfig
from supervised_product_matching.model_preprocessing import bert_preprocess_batch

class SiameseNetwork(nn.Module):
    def __init__(self, h_size=768):
        '''
        Model that uses BERT to classify the titles.
        h_size: The hidden layer size for the classification token (CLS) in BERT (Default: 768)
        '''

        super(SiameseNetwork, self).__init__()
        self.h_size = h_size
        
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

    def forward(self, input1, input2):
        '''
        x is going to be a numpy array of [sentenceA, sentenceB].
        Model using BERT to make a prediction of whether the two titles represent 
        the same entity.
        '''

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

def forward_prop(batch_data, batch_labels, net, criterion):
    # Forward propagation
    forward = net(*bert_preprocess_batch(batch_data))

    # Convert batch labels to Tensor
    batch_labels = torch.from_numpy(batch_labels).view(-1).long().to(ModelConfig.device)

    # Calculate loss
    loss = criterion(forward, batch_labels)

    return loss, forward

