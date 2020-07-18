import torch
import torch.nn as nn

''' 
position-wise feed-forward network

attributes:
    d_model: size of word embeddings
    hidden_size: size of position-wise feed forward network
    dropout: dropout
'''
class PositionWiseFeedForward(nn.Module):

    def __init__(self, d_model, hidden_size, dropout = 0.5):
        super().__init__()

        self.W_1 = nn.Linear(d_model, hidden_size)
        self.W_2 = nn.Linear(hidden_size, d_model)

        self.layer_norm = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    '''
    input param:
        x: output of multi-head self-attention network (batch_size, word_pad_len, d_model)

    return: 
        out: output of position-wise feed-forward network (batch_size, word_pad_len, d_model)
    '''
    def forward(self, x):
        # eq.2: FFN = max(0, x W_1 + b_1) W_2 + b_2
        out = self.W_2(self.relu(self.W_1(x)))  # (batch_size, word_pad_len, d_model)
        out = self.dropout(out)

        out += x  # residual connection
        out = self.layer_norm(out)  # LayerNorm

        return out