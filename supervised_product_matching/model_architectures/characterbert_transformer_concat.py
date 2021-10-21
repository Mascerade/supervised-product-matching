# CharacterBERT citation for authors:
'''
Paper Title: CharacterBERT: Reconciling ELMo and BERT for Word-Level Open-Vocabulary Representations From Characters
Authors: Hicham El Boukkouri and Olivier Ferret and Thomas Lavergne and Hiroshi Noji and Pierre Zweigenbaum and Junichi Tsujii
The characterbert_modeling and characterbert_utils were also created by them
Their GitHub Repo is at: https://github.com/helboukkouri/character-bert
'''
import torch
import torch.nn as nn
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
        self.sequence_length = ModelConfig.max_len * 2 + 3
        self.h_size = h_size
        
        # CharacterBERT model
        self.bert = CharacterBertModel.from_pretrained('./pretrained-models/general_character_bert/')

        # Define the Scaling Layers
        self.scale1 = ScalingLayer(in_features=h_size, out_features=384, pwff_inner_features=2048, pwff_dropout=0.5)
        self.scale2 = ScalingLayer(in_features=384, out_features=32, pwff_inner_features=768, pwff_dropout=0.5)
        
        # Dropout for overfitting
        self.dropout_5 = nn.Dropout(p=0.5)

        # Dropout last
        self.dropout_7 = nn.Dropout(p=0.7)

        # Linear layer for classification'
        self.classification = nn.Linear(in_features=self.sequence_length * 32, out_features=2)
        
        # Softmax for prediction
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input1, input2):
        '''
        x is going to be a numpy array of [sentenceA, sentenceB].
        Model using CharacterBERT to make a prediction of whether the two titles represent 
        the same entity.
        '''
        # Get the amount of batches from the input
        batches = input1.size()[0]

        # Send the inputs through BERT
        # We index at 0 because that gives us the output for each token
        output1 = self.bert(input1)[0]
        output2 = self.bert(input2)[0]

        # BERT calls for the addition of both 
        addition = output1 + output2

        # Dropout
        addition = self.dropout_5(addition)
        
        # Forward propagate through first scaled Transformer
        scaled = self.scale1(addition)
        
        # Dropout
        scaled = self.dropout_5(scaled)
        
        # Forward propagate through second scaled Transformer
        scaled = self.scale2(scaled)
        scaled = scaled.view(batches, -1)
        
        # Dropout
        scaled = self.dropout_7(scaled)

        # Go through final linear layer
        out = self.classification(scaled)

        # Dropout
        out = self.dropout_7(out)

        # Softmax Activation to get predictions
        out = self.softmax(out)

        return out

def forward_prop(batch_data, batch_labels, net, criterion):
    # Forward propagation
    forward = net(*character_bert_preprocess_batch(batch_data, pad=True))

    # Convert batch labels to Tensor
    batch_labels = torch.from_numpy(batch_labels).view(-1).long().to(ModelConfig.device)

    # Calculate loss
    loss = criterion(forward, batch_labels)

    # Add L2 Regularization to the Transformers and final linear layer
    l2_lambda = 2e-3
    l2_reg = torch.tensor(0.)
    for param in net.scale1.parameters():
        l2_reg += torch.norm(param)
    for param in net.scale2.parameters():
        l2_reg += torch.norm(param)
    for param in net.classification.parameters():
        l2_reg += torch.norm(param)

    # Add L2 Regularization to bert
    l2_lambda_bert = 7e-5
    l2_reg_bert = torch.tensor(0.)
    for param in net.bert.parameters():
        l2_reg_bert += torch.norm(param)

    loss += l2_lambda * l2_reg + l2_lambda_bert * l2_reg_bert

    return loss, forward
