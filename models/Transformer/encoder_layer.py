import torch
import torch.nn as nn
from .attention import MultiHeadAttention
from .ffn import PositionWiseFeedForward

'''
an encoder layer

attributes:
    d_model: size of word embeddings
    n_heads: number of attention heads
    hidden_size: size of position-wise feed forward network
    dropout: dropout
'''
class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, hidden_size, dropout = 0.5):
        super(EncoderLayer, self).__init__()
        
        # an encoder layer has two sub-layers: 
        #   - multi-head self-attention
        #   - positon-wise fully connected feed-forward network
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, hidden_size, dropout)

    '''
    input param:
        x: input data (batch_size, word_pad_len, d_model)
        mask: padding mask metrix, None if it is not needed (batch_size, 1, word_pad_len)
    return:
        out: output of the current encoder layer (batch_size, word_pad_len, d_model)
        att: attention weights (batch_size, n_heads, word_pad_len, word_pad_len)
    '''
    def forward(self, x, mask = None):
        att_out, att = self.attention(x, mask = mask)  # (batch_size, word_pad_len, d_model), (batch_size, n_heads, word_pad_len, word_pad_len)
        out = self.feed_forward(att_out)  # (batch_size, word_pad_len, d_model)
        return out, att