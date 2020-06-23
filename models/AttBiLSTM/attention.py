import torch
from torch import nn

'''
attention network

attributes:
    rnn_size: size of bi-LSTM
'''
class Attention(nn.Module):
    def __init__(self, rnn_size):
        super(Attention, self).__init__()
        self.w = nn.Linear(rnn_size, 1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim = 1)
    
    '''
    input param: 
        H: output of bi-LSTM (batch_size, word_pad_len, hidden_size)
    
    return:
        r: sentence representation r (batch_size, rnn_size)
        alpha: attention weights (batch_size, word_pad_len)
    '''
    def forward(self, H):

        # eq.9: M = tanh(H)
        M = self.tanh(H)  # (batch_size, word_pad_len, rnn_size)
        
        # eq.10: α = softmax(w^T M)
        alpha = self.w(M).squeeze(2)  # (batch_size, word_pad_len)
        alpha = self.softmax(alpha)  # (batch_size, word_pad_len)
        
        # eq.11: r = H 
        r = H * alpha.unsqueeze(2)  # (batch_size, word_pad_len, rnn_size)
        r = r.sum(dim = 1)  # (batch_size, rnn_size)
        
        return r, alpha