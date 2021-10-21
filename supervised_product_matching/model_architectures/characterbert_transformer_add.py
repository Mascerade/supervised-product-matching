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
from character_bert.modeling.character_bert import CharacterBertModel
from scale_transformer_encoder.scaling_layer import ScalingLayer
from supervised_product_matching.config import ModelConfig
from supervised_product_matching.model_preprocessing import character_bert_preprocess_batch

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

        # Define the Scaling Layers
        self.scale1 = ScalingLayer(in_features=h_size, out_features=512, pwff_inner_features=2048, pwff_dropout=0.1)
        self.scale2 = ScalingLayer(in_features=512, out_features=256, pwff_inner_features=1028, pwff_dropout=0.1)

        # Dropout layers
        self.dropout_1 = nn.Dropout(p=0.1)
        self.dropout_2 = nn.Dropout(p=0.2)
        self.dropout_3 = nn.Dropout(p=0.3)
        self.dropout_4 = nn.Dropout(p=0.4)
        self.dropout_5 = nn.Dropout(p=0.5)
        self.dropout_7 = nn.Dropout(p=0.7)

        # Linear layer for classification'
        self.classification = nn.Linear(in_features=256, out_features=2)
        
        # Softmax for prediction
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input1, input2):
        '''
        x is going to be a numpy array of the sequences
        Model using CharacterBERT to make a prediction of whether the two titles represent 
        the same entity.
        '''
        # Get the amount of batches from the input
        sequence_length = input1.size()[1]

        # Send the inputs through BERT
        # We index at 0 because that gives us the output for each token
        bert_output1 = self.bert(input1)[0]
        bert_output2 = self.bert(input2)[0]

        # Dropout
        bert_output1 = self.dropout_1(bert_output1)
        bert_output2 = self.dropout_1(bert_output1)

        # Use the first Transformer on each output
        scaled1 = self.scale1(bert_output1)
        scaled2 = self.scale1(bert_output2)

        # Dropout
        scaled1 = self.dropout_1(scaled1)
        scaled2 = self.dropout_1(scaled2)

        # Use the second Transformer on each output
        scaled1 = self.scale2(scaled1)
        scaled2 = self.scale2(scaled2)

        # Dropout
        scaled = scaled1 + scaled2
        
        # Average token embeddings
        scaled = scaled[:, 1:].sum(dim=1) / sequence_length
        
        # Dropout
        scaled = self.dropout_5(scaled)

        # Go through final linear layer
        out = self.classification(scaled)

        # Dropout
        out = self.dropout_5(out)

        # Softmax Activation to get predictions
        out = self.softmax(out)

        return out

def forward_prop(batch_data, batch_labels, net, criterion):
    # Forward propagation
    forward = net(*character_bert_preprocess_batch(batch_data, pad=False))

    # Convert batch labels to Tensor
    batch_labels = torch.from_numpy(batch_labels).view(-1).long().to(ModelConfig.device)

    # Calculate loss
    loss = criterion(forward, batch_labels).to(ModelConfig.device)

    # Add L2 Regularization to the Transformers and final linear layer
    l2_lambda_scale = 5e-4
    l2_lambda_linear = 5e-1
    l2_reg_scale = torch.tensor(0.).to(ModelConfig.device)
    l2_reg_linear = torch.tensor(0.).to(ModelConfig.device)
    for param in net.scale1.parameters():
        l2_reg_scale += torch.norm(param)
    for param in net.scale2.parameters():
        l2_reg_scale += torch.norm(param)
    for param in net.classification.parameters():
        l2_reg_linear += torch.norm(param)

    # Add L2 Regularization to bert
    l2_lambda_bert = 5e-5
    l2_reg_bert = torch.tensor(0.).to(ModelConfig.device)
    for param in net.bert.parameters():
        l2_reg_bert += torch.norm(param)

    loss += l2_lambda_scale * l2_reg_scale + l2_lambda_linear * l2_reg_linear + l2_lambda_bert * l2_reg_bert

    return loss, forward
