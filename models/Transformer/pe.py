import torch
import torch.nn as nn
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''
positional encoding

attributes:
    d_model: size of word embeddings
    word_pad_len: length of padded sentence
    dropout: dropout
'''
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, word_pad_len, dropout):
        super(PositionalEncoding, self).__init__()

        self.pe = torch.tensor([
            [pos / (10000.0 ** (i // 2 * 2.0 / d_model)) for i in range(d_model)] 
            for pos in range(word_pad_len)
        ])  # (batch_size, word_pad_len, emb_size)

        # PE(pos, 2i) = sin(pos / 10000^{2i / d_model})
        self.pe[:, 0::2] = np.sin(self.pe[:, 0::2])
        # PE(pos, 2i + 1) = cos(pos / 10000^{2i / d_model})
        self.pe[:, 1::2] = np.cos(self.pe[:, 1::2])
        
        self.dropout = nn.Dropout(dropout)

    '''
    input param:
        embeddings: word embeddings (batch_size, word_pad_len, emb_size)

    return: 
        word embeddings + positional encoding (batch_size, word_pad_len, emb_size)
    '''
    def forward(self, embeddings):
        # word embeddings + positional encoding
        embeddings = embeddings + nn.Parameter(self.pe, requires_grad = False).to(device)
        embeddings = self.dropout(embeddings)
        return embeddings